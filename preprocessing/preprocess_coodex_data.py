import pandas as pd
import numpy as np
import openslide 
import tifffile as tif
import torch
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import cv2
import zarr
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, square
import SimpleITK as sitk
import histomicstk as htk


def min_area_rect_center_angle(mask: np.ndarray):
    """
    Finds the largest contour in 'mask' and returns a cv2.minAreaRect:
      - center (x, y)
      - (width, height)
      - angle (degrees)
    Returns (None, None, None) if no valid contour found.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None, None, None
    (cx, cy), (w, h), angle = cv2.minAreaRect(largest)
    return (cx, cy), (w, h), angle


def build_rotate_translate_transform(
    src_center: tuple, src_angle: float,
    dst_center: tuple, dst_angle: float
) -> np.ndarray:
    """
    Build a 3x3 affine matrix that:
      1) Rotates around src_center by (dst_angle - src_angle).
      2) Translates so that src_center => dst_center.
    
    No scaling of widths/heights is done. Only rotation+translation.

    Returns
    -------
    M : np.ndarray
        3x3 transform mapping source -> destination.
    """
    (cx_s, cy_s) = src_center
    (cx_d, cy_d) = dst_center
    delta_angle = dst_angle - src_angle
    theta = np.deg2rad(delta_angle)

    # 1) Rotate around (cx_s, cy_s)
    #    M_rot = T(+cx_s, +cy_s) * R(theta) * T(-cx_s, -cy_s)
    Tneg = np.array([[1, 0, -cx_s],
                     [0, 1, -cy_s],
                     [0, 0,    1  ]], dtype=np.float32)
    R = np.array([[ np.cos(theta), -np.sin(theta), 0],
                  [ np.sin(theta),  np.cos(theta), 0],
                  [         0,              0,     1]], dtype=np.float32)
    Tpos = np.array([[1, 0, cx_s],
                     [0, 1, cy_s],
                     [0, 0,   1 ]], dtype=np.float32)
    M_rot = Tpos @ R @ Tneg

    # 2) Translate so that the new center_s => center_d
    #    After rotation, the center remains (cx_s, cy_s). 
    tx = cx_d - cx_s
    ty = cy_d - cy_s
    Tfinal = np.array([[1, 0, tx],
                       [0, 1, ty],
                       [0, 0,  1 ]], dtype=np.float32)

    # Full transform
    M = Tfinal @ M_rot
    return M


def extract_grayscale_masks_downsample(he_rgba, fluo, scale=0.1, close_size=5):
    """
    Downsample H&E and fluo. Return the downsampled images + their binary masks.
    """
    # 1) Convert RGBA->RGB if needed
    if he_rgba.shape[-1] == 4:
        he_rgb = he_rgba[..., :3]
    else:
        he_rgb = he_rgba

    # 2) Move fluo => (H, W, C)
    fluo_t = np.transpose(fluo, (1, 2, 0))

    # 3) Downsample
    new_h_he = int(he_rgb.shape[0] * scale)
    new_w_he = int(he_rgb.shape[1] * scale)
    new_h_fluo = int(fluo_t.shape[0] * scale)
    new_w_fluo = int(fluo_t.shape[1] * scale)

    he_t = torch.Tensor(he_rgb.astype('float32')).permute(2,0,1).unsqueeze(0).float()
    he_small_t = Resize((new_h_he, new_w_he))(he_t)
    he_small = he_small_t.squeeze(0).permute(1,2,0).numpy()

    fluo_ft = torch.Tensor(fluo_t.astype('float32')).permute(2,0,1).unsqueeze(0).float()
    fluo_small_t = Resize((new_h_fluo, new_w_fluo))(fluo_ft)
    fluo_small = fluo_small_t.squeeze(0).permute(1,2,0).numpy()

    # 4) Create masks
    he_gray_small = np.dot(he_small[..., :3], [0.299, 0.587, 0.114])
    he_gray_small /= (he_gray_small.max() + 1e-9)
    he_gray_small = 1 - he_gray_small  # invert so tissue is bright

    fluo_norm_small = fluo_small.copy()
    for c in range(fluo_norm_small.shape[-1]):
        mn, mx = fluo_norm_small[..., c].min(), fluo_norm_small[..., c].max()
        fluo_norm_small[..., c] = (fluo_norm_small[..., c]-mn)/(mx-mn+1e-9)
    fluo_gray_small = np.max(fluo_norm_small, axis=-1)

    th_he = threshold_otsu(he_gray_small)
    mask_he_small = (he_gray_small > th_he)
    mask_he_small = binary_closing(mask_he_small, square(close_size))

    th_fluo = threshold_otsu(fluo_gray_small)
    mask_fluo_small = (fluo_gray_small > th_fluo)
    mask_fluo_small = binary_closing(mask_fluo_small, square(close_size))

    return he_small, fluo_small, mask_he_small, mask_fluo_small


def warp_full_resolution_fluo(
    fluo_full_trans: np.ndarray,
    M_full: np.ndarray
) -> np.ndarray:
    """
    Warp the full-resolution fluo (H,W,C) with a 3x3 Affine-like transform
    (we can use warpPerspective). Return (H,W,C).
    """
    H, W = fluo_full_trans.shape[:2]
    warped = cv2.warpPerspective(
        fluo_full_trans, M_full, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped


def crop_or_pad_same_size(imgA: np.ndarray, imgB: np.ndarray):
    """
    Make sure 'imgA' and 'imgB' have the same shape by creating a union bounding box
    and placing each image at its (0,0). But here, since we want final shapes to match
    AND we want them aligned by centroid, we can do something simpler:
    
    We'll just pick max(H,W). This is a simplified approach if both images
    are the same resolution. If not, you can do a bounding box approach.

    Returns
    -------
    outA, outB : np.ndarray
    """
    Ha, Wa = imgA.shape[:2]
    Hb, Wb = imgB.shape[:2]
    H_out = max(Ha, Hb)
    W_out = max(Wa, Wb)

    def pad(img, H, W):
        if img.ndim == 2:
            out = np.zeros((H, W), dtype=img.dtype)
            out[:img.shape[0], :img.shape[1]] = img
        else:
            c = img.shape[2] if img.ndim==3 else 1
            if c==1:
                out = np.zeros((H, W), dtype=img.dtype)
                out[:img.shape[0], :img.shape[1]] = img[...,0]
                return out
            out = np.zeros((H, W, c), dtype=img.dtype)
            out[:img.shape[0], :img.shape[1], :] = img
        return out

    outA = pad(imgA, H_out, W_out)
    outB = pad(imgB, H_out, W_out)
    return outA, outB


def align_he_fluo_by_bounding_box(
    he_rgba: np.ndarray,
    fluo: np.ndarray,
    scale: float = 0.1,
    close_size: int = 5
):
    """
    1. Downsample H&E + fluo
    2. Otsu => produce masks
    3. Use minAreaRect to get (center, angle) for each mask
    4. Build a rotate+translate (no scale) to align fluo bounding box to H&E bounding box
    5. Scale that transform up => M_full
    6. Warp the *full-res* fluo
    7. Crop/Pad the final images so they have the same shape & aligned centroids.

    Returns
    -------
    aligned_he : np.ndarray
        Full-resolution H&E (unchanged, but possibly padded)
    aligned_fluo : np.ndarray
        Full-resolution fluo after bounding-box alignment + padding
    """
    print("=== Step 1: Downsample & get masks ===")
    he_small, fluo_small, mask_he_small, mask_fluo_small = extract_grayscale_masks_downsample(
        he_rgba, fluo, scale=scale, close_size=close_size
    )

    print("=== Step 2: minAreaRect => (center, angle) ===")
    (cx_he, cy_he), (w_he, h_he), angle_he = min_area_rect_center_angle(mask_he_small)
    (cx_fluo, cy_fluo), (w_fluo, h_fluo), angle_fluo = min_area_rect_center_angle(mask_fluo_small)

    if cx_he is None or cx_fluo is None:
        print("WARNING: Could not get bounding box. Using identity transform.")
        M_small = np.eye(3, dtype=np.float32)
    else:
        # Build rotate+translate in downsample space
        M_small = build_rotate_translate_transform(
            src_center=(cx_fluo, cy_fluo), src_angle=angle_fluo,
            dst_center=(cx_he, cy_he),   dst_angle=angle_he
        )

    print("=== Step 3: Scale M_small => M_full for full resolution ===")
    S = np.array([
        [scale,   0,     0],
        [  0,   scale,   0],
        [  0,     0,     1]
    ], dtype=np.float32)
    S_inv = np.linalg.inv(S)
    M_full = S_inv @ M_small @ S

    # Convert fluo to (H,W,C)
    fluo_full_trans = np.transpose(fluo, (1, 2, 0))

    print("=== Step 4: Warp full-res fluo with M_full ===")
    warped_fluo_full = warp_full_resolution_fluo(fluo_full_trans, M_full)

    print("=== Step 5: Ensure final H&E & fluo have same shape ===")
    # H&E is reference, but we might pad it if fluo ended up bigger after warp
    # We won't warp H&E, so the tissue remains in the original coordinate system.
    he_rgb_full = he_rgba[..., :3] if he_rgba.shape[-1]==4 else he_rgba

    aligned_he, aligned_fluo = crop_or_pad_same_size(he_rgb_full, warped_fluo_full)

    return aligned_he, aligned_fluo


def mean_square_difference_registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    min_step: float = 1e-10,
    sampling_percentage: float = 0.9,
    max_iterations: int = 100,
    learning_rate: float = 100.0,
    output_transform_file: str = None
) -> tuple:
    """
    Perform mean square difference-based image registration between two aligned images.

    Parameters
    ----------
    fixed_image : np.ndarray
        Fixed image (e.g., H&E image), shape [Y, X].
    moving_image : np.ndarray
        Moving image (e.g., fluorescence image), shape [Y, X].
    sampling_percentage : float, optional
        Percentage of random points to sample for metric calculation. Default=0.50.
    max_iterations : int, optional
        Maximum number of optimizer iterations. Default=100.
    learning_rate : float, optional
        Initial step size for the optimizer. Default=100.0.
    output_transform_file : str, optional
        Path to save the resulting transform file. Default=None.

    Returns
    -------
    registered_image : np.ndarray
        The moving image warped to match the fixed image.
    transform : sitk.Transform
        The final transform object.
    resampler : sitk.ResampleImageFilter
        The resampler used for the registration.
    """

    def command_iteration(registration_method):
        """Callback invoked during optimization iterations."""
        print(
            f"Iteration {registration_method.GetOptimizerIteration()} "
            + f"Metric value: {registration_method.GetMetricValue():.5f}"
        )

    # Convert NumPy arrays to SimpleITK images
    fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))

    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Set metric as mean square difference
    registration_method.SetMetricAsMeanSquares()

    # Optional: Set random sampling
    registration_method.SetMetricSamplingPercentage(sampling_percentage, sitk.sitkWallClock)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)

    # Optimizer settings
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=max_iterations,
        relaxationFactor=0.5
    )

    # Set the initial transform as a translation
    initial_transform = sitk.TranslationTransform(fixed_sitk.GetDimension())
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Attach the iteration callback
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    # Execute the registration
    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

    print("Registration Complete!")
    print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f" Iterations: {registration_method.GetOptimizerIteration()}")
    print(f" Final metric value: {registration_method.GetMetricValue()}")

    # Save the transform if requested
    if output_transform_file:
        sitk.WriteTransform(final_transform, output_transform_file)

    # Resample the moving image to align with the fixed image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)  # Background pixel value
    resampler.SetTransform(final_transform)

    registered_sitk = resampler.Execute(moving_sitk)
    registered_image = sitk.GetArrayFromImage(registered_sitk)

    return registered_image, final_transform, resampler


def bspline_correlation_registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    transform_domain_mesh_size: int = 8,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-5,
    function_tolerance: float = 1e-7,
    cost_function_factor: float = 1e7,
    output_transform_file: str = None
) -> tuple:
    """
    Perform B-spline-based image registration using the complex correlation metric.

    Parameters
    ----------
    fixed_image : np.ndarray
        Fixed image (e.g., H&E image), shape [Y, X].
    moving_image : np.ndarray
        Moving image (e.g., fluorescence image), shape [Y, X].
    transform_domain_mesh_size : int, optional
        Number of control points for the B-spline grid. Default=8.
    max_iterations : int, optional
        Maximum number of iterations for the optimizer. Default=100.
    gradient_tolerance : float, optional
        Tolerance for convergence based on gradient norm. Default=1e-5.
    function_tolerance : float, optional
        Convergence factor for the cost function. Default=1e-7.
    cost_function_factor : float, optional
        Scaling factor for cost function convergence. Default=1e7.
    output_transform_file : str, optional
        Path to save the resulting transform file. Default=None.

    Returns
    -------
    registered_image : np.ndarray
        The moving image warped to match the fixed image.
    transform : sitk.Transform
        The final B-spline transform object.
    """

    def command_iteration(registration_method):
        """Callback invoked during optimization iterations."""
        print(
            f"Iteration {registration_method.GetOptimizerIteration()} "
            + f"Metric value: {registration_method.GetMetricValue():.5f}"
        )

    # Convert NumPy arrays to SimpleITK images
    fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))

    # Initialize the B-spline transform
    transform_mesh_size = [transform_domain_mesh_size] * fixed_sitk.GetDimension()
    initial_transform = sitk.BSplineTransformInitializer(fixed_sitk, transform_mesh_size)

    print("Initial Transform Parameters:")
    print(initial_transform.GetParameters())

    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Set metric as correlation
    registration_method.SetMetricAsCorrelation()

    # Optimizer settings
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=gradient_tolerance,
        numberOfIterations=max_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=cost_function_factor,
    )

    # Use the B-spline transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Attach the iteration callback
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    # Execute the registration
    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

    print("Registration Complete!")
    print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f" Iterations: {registration_method.GetOptimizerIteration()}")
    print(f" Final metric value: {registration_method.GetMetricValue()}")

    # Save the transform if requested
    if output_transform_file:
        sitk.WriteTransform(final_transform, output_transform_file)

    # Resample the moving image to align with the fixed image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)  # Background pixel value
    resampler.SetTransform(final_transform)

    registered_sitk = resampler.Execute(moving_sitk)
    registered_image = sitk.GetArrayFromImage(registered_sitk)

    return registered_image, final_transform, resampler


def apply_resampler_transform(
    resampler: sitk.ResampleImageFilter,
    reference_image: np.ndarray,
    moving_image: np.ndarray
) -> np.ndarray:
    """
    Apply a learned resampler transform to a new moving image.

    Parameters
    ----------
    resampler : sitk.ResampleImageFilter
        The resampler object containing the learned transform.
    reference_image : np.ndarray
        The reference (fixed) image used for registration, shape [Y, X].
    moving_image : np.ndarray
        The new moving image to transform, shape [Y, X].

    Returns
    -------
    transformed_image : np.ndarray
        The transformed moving image aligned to the reference image.
    """
    # Convert the reference and moving images to SimpleITK images
    reference_sitk = sitk.GetImageFromArray(reference_image.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))

    # Get the transform from the resampler
    learned_transform = resampler.GetTransform()

    # Create a new resampler for the new moving image
    new_resampler = sitk.ResampleImageFilter()
    new_resampler.SetReferenceImage(reference_sitk)
    new_resampler.SetTransform(learned_transform)
    new_resampler.SetInterpolator(sitk.sitkLinear)
    new_resampler.SetDefaultPixelValue(0)  # Set background value for out-of-bounds regions

    # Apply the transform to the new moving image
    transformed_sitk = new_resampler.Execute(moving_sitk)
    transformed_image = sitk.GetArrayFromImage(transformed_sitk)

    return transformed_image


def resize_image(image, scale_factor):
    image_tensor = torch.tensor(image.astype('float32'))
    image_tensor = image_tensor.permute(2, 0, 1)  # shape: [Channel, Y, X]
    image_tensor = image_tensor.unsqueeze(0)  # shape: [Batch, Channel, Y, X]
    resize_tranform = Resize((int(scale_factor * image.shape[0]), int(scale_factor * image.shape[1])))
    resized_image = resize_tranform(image_tensor)
    resized_image = resized_image.squeeze(0).permute(1, 2, 0).numpy()
    return resized_image


def align(he_file_path, codex_file_path, idx, output_path):
    slide_he = openslide.OpenSlide(he_file_path)
    slide_codex = tif.imread(codex_file_path)
    slide_codex = np.array(slide_codex).astype(np.float32)
    slide_codex = np.flip(slide_codex, axis=2)
    full_im = slide_he.read_region((0,0), 0, slide_he.level_dimensions[0])
    full_im = torch.Tensor(np.array(full_im))  # shape: [H, W, C]
    full_im = full_im.permute(1, 0, 2)  # shape: [W, H, C]
    full_im = full_im.numpy().astype('float32')  # shape: [W, H, C]
    full_he_image, full_fluo_image = align_he_fluo_by_bounding_box(
                                he_rgba=full_im,
                                fluo=slide_codex,
                                scale=0.1,
                                close_size=30)
    
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    stains = ['hematoxylin',  # nuclei stain
            'eosin',        # cytoplasm stain
            'null']         # set to null if input contains only two stains
    W = np.array([stain_color_map[st] for st in stains]).T
    full_he_decon = htk.preprocessing.color_deconvolution.color_deconvolution(full_he_image, W)
    full_hemotox_image = full_he_decon.Stains[:, :, 0]

    # performing first alignment
    full_fluo_dapi = np.transpose(full_fluo_image, (2,0,1))[0]
    full_fluo_dapi = full_fluo_dapi / np.amax(full_fluo_dapi)
    full_fluo_dapi *= 255

    full_hemotox_inverted = 255 - full_hemotox_image
    registered_image_1, _, linear_transform_itk = mean_square_difference_registration(full_hemotox_inverted,
        full_fluo_dapi,
        min_step=1e-12,
        max_iterations = 100,
        learning_rate = 200)

    # performing second alignment with B-spline
    registered_image_2, _, bspline_transform_itk = bspline_correlation_registration(
        fixed_image=full_hemotox_inverted,
        moving_image=registered_image_1,
        transform_domain_mesh_size=8,
        max_iterations=100,
        gradient_tolerance=1e-7,
        function_tolerance=1e-7,
        cost_function_factor=1e7
    )

    aligned_fluo_chan_first = np.transpose(full_fluo_image, (2, 0, 1))
    aligned_channels = [registered_image_2]
    for chan in aligned_fluo_chan_first:
        chan = chan.astype('float32') # shape [H, W]
        # putting through the first
        chan_t1 = apply_resampler_transform(linear_transform_itk, full_hemotox_inverted, chan)
        chan_t2 = apply_resampler_transform(bspline_transform_itk, full_hemotox_inverted, chan_t1)
        aligned_channels.append(chan_t2)
    aligned_codex = np.array(aligned_channels)
    if output_path.endswith('/'):
        output_path = output_path[:-1]
    
    he_file_path = f'{output_path}/aligned_hemotox_{idx}.ome.tif'
    codex_file_path = f'{output_path}/aligned_codex_{idx}.ome.tif'
    tif.imwrite(he_file_path, full_he_image)
    tif.imwrite(codex_file_path, aligned_codex)
    return he_file_path, codex_file_path


def process_wsi(image1, image2, tile_size, min_tissue_percentage=0.1, visualize_patches=False):
    """
    Process two full-size WSI images to extract KxK tiles containing tissue from both images,
    by dividing the entire image into patches and filtering based on the mask.

    Parameters
    ----------
    image1 : np.ndarray
        Full-size WSI image of shape (Y, X, 3) used to generate the mask.
    image2 : np.ndarray
        Full-size WSI image of shape (Y, X, N) from which tiles will be extracted.
    tile_size : int
        Size of the KxK tiles to extract.
    min_tissue_percentage : float, optional
        Minimum percentage of the tile that must contain tissue to be included.
        Default is 0.5 (50%).
    visualize_patches : bool, optional
        If True, displays the downsampled image with patches overlaid as either kept or blacked out.

    Returns
    -------
    tiles1 : list of np.ndarray
        List of KxK image tiles from image1 containing tissue.
    tiles2 : list of np.ndarray
        List of KxK image tiles from image2 corresponding to the same regions.
    locations : list of tuple
        List of (row, col) coordinates of the top-left corners of the kept tiles.
    """
    # Step 1: Convert the first image to grayscale
    image1 = image1.astype(np.uint8)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    # Step 2: Downsample the image by a factor of 10
    downsample_factor = 10
    downsampled_image1 = cv2.resize(
        gray_image1, 
        (gray_image1.shape[1] // downsample_factor, gray_image1.shape[0] // downsample_factor),
        interpolation=cv2.INTER_AREA
    )

    # Step 3: Invert the downsampled image
    inverted_image1 = 255 - downsampled_image1

    # Step 4: Apply Otsu's thresholding to generate a binary mask
    otsu_thresh = threshold_otsu(inverted_image1)
    binary_mask = (inverted_image1 > otsu_thresh).astype(np.uint8)

    # Initialize visualization overlay
    patch_overlay = np.zeros_like(downsampled_image1)

    # Step 5: Divide the original image into patches and evaluate them
    tiles1 = []
    tiles2 = []
    locations = []

    rows, cols = image1.shape[0], image1.shape[1]
    for i in range(0, rows, tile_size):
        for j in range(0, cols, tile_size):
            # Ensure the tile is within image bounds
            if i + tile_size <= rows and j + tile_size <= cols:
                # Extract the tile from both images
                tile1 = image1[i:i + tile_size, j:j + tile_size]
                tile2 = image2[i:i + tile_size, j:j + tile_size]

                # Adjust the channels of image2
                if tile2.shape[-1] == 14:
                    tile2 = tile2[:, :, 1:]
                elif tile2.shape[-1] == 34:
                    tile2 = tile2[:, :, [1, 25, 23, 15, 19, 26, 27, 28, 5, 30, 31, 29, 32]]
                else:
                    raise ValueError("Invalid number of channels in image2")

                # Compute the tissue percentage using the downsampled binary mask
                downsampled_tile_mask = binary_mask[
                    i // downsample_factor:(i + tile_size) // downsample_factor,
                    j // downsample_factor:(j + tile_size) // downsample_factor
                ]
                tissue_area = np.sum(downsampled_tile_mask)
                total_area = downsampled_tile_mask.size
                tissue_percentage = tissue_area / total_area

                # Keep the tile if it meets the tissue threshold
                if tissue_percentage >= min_tissue_percentage:
                    tiles1.append(tile1)
                    tiles2.append(tile2)
                    locations.append((i, j))
                    
                    # Mark the patch as kept in the overlay
                    patch_overlay[
                        i // downsample_factor:(i + tile_size) // downsample_factor,
                        j // downsample_factor:(j + tile_size) // downsample_factor
                    ] = downsampled_image1[
                        i // downsample_factor:(i + tile_size) // downsample_factor,
                        j // downsample_factor:(j + tile_size) // downsample_factor
                    ]

    # Optional: Visualize patches on the downsampled image
    if visualize_patches:
        plt.figure(figsize=(10, 10))
        plt.imshow(patch_overlay, cmap='gray')
        plt.title("Patches Visualization")
        plt.axis('off')
        plt.show()

    assert len(tiles1) == len(tiles2) == len(locations)
    return tiles1, tiles2, locations
    
    
def main():
    # Setting up Argparse
    parser = ArgumentParser()
    parser.add_argument('--he_file_list', type=str, required=True)
    parser.add_argument('--codex_file_list', type=str, required=True)
    parser.add_argument('--zarr_filename', type=str, required=True)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--min_tissue_percentage', type=float, default=0.1)
    parser.add_argument('--visualize_patches', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Reading in the file lists
    aligned_he_file_list = []
    aligned_codex_file_list = []
    he_file_list = pd.read_csv(args.he_file_list)
    codex_file_list = pd.read_csv(args.codex_file_list)

    # Performing Alignment
    print("Performing alignment...")
    for he_file, codex_file, idx in tqdm(zip(he_file_list['file_path'], codex_file_list['file_path'], range(len(he_file_list))), total=len(he_file_list)):
        he_fn, codex_fn = align(he_file, codex_file, idx, args.output_dir)
        aligned_he_file_list.append(he_fn)
        aligned_codex_file_list.append(codex_fn)
    
    # Setting up Zarr File
    output_dir = args.output_dir
    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    zarr_file = f'{output_dir}/{args.zarr_filename}.zarr'
    zarr_root = zarr.open(zarr_file, mode='w')

    # Extractinmg the tiles and saving to Zarr
    print("Extracting tiles and saving to Zarr...")
    for he_file_aligned, codex_file_aligned, idx in tqdm(zip(aligned_he_file_list, aligned_codex_file_list, range(len(aligned_he_file_list))), total=len(aligned_he_file_list)):
        grp = zarr_root.create_group(f'aligned_{idx}')
        he = tif.imread(he_file_aligned)
        codex = tif.imread(codex_file_aligned)
        tiles1, tiles2, locations = process_wsi(he, codex.transpose(1,2,0), tile_size=args.tile_size, min_tissue_percentage=args.min_tissue_percentage, visualize_patches=args.visualize_patches)
        tiles1 = np.array(tiles1)
        tiles2 = np.array(tiles2)
        locations = np.array(locations)
        grp.create_dataset("he", data=tiles1, chunks=(1, 3, args.tile_size, args.tile_size))
        grp.create_dataset("codex", data=tiles2, chunks=(1, tiles2.shape[1], args.tile_size, args.tile_size))
        grp.create_dataset("locations", data=locations)
    
    # saving the original and aligned file paths to csv file
    aligned_file_list = pd.DataFrame({'aligned_he_file_path': aligned_he_file_list, 'aligned_codex_file_path': aligned_codex_file_list})
    original_file_list = pd.DataFrame({'original_he_file_path': he_file_list['file_path'], 'original_codex_file_path': codex_file_list['file_path']})
    combined_file_list = pd.concat([original_file_list, aligned_file_list], axis=1)
    combined_file_list.to_csv(f'{output_dir}/aligned_file_list.csv', index=False)
    print(f"Saved aligned file list to {output_dir}/aligned_file_list.csv")
    print(f"Saved aligned data to {zarr_file}")
    print("Done!")


if __name__ == '__main__':
    main()

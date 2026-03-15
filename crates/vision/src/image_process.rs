//! Image preprocessing: smart_resize, normalize, patchify.

use herbert_core::error::{HerbertError, Result};

use crate::config::VisionConfig;

/// Compute grid-aligned resize dimensions.
///
/// Matches HuggingFace `smart_resize`: rounds to nearest `grid_factor` multiple,
/// then clamps to `[min_pixels, max_pixels]` total pixel budget.
pub fn smart_resize(
    height: usize,
    width: usize,
    patch_size: usize,
    merge_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let (h, w) = (height as f64, width as f64);
    let max_dim = h.max(w);
    let min_dim = h.min(w);
    if min_dim <= 0.0 {
        return Err(HerbertError::Config("image dimensions must be > 0".into()));
    }
    if max_dim / min_dim > 200.0 {
        return Err(HerbertError::Config(format!(
            "aspect ratio must be < 200, got {:.1}",
            max_dim / min_dim
        )));
    }

    let grid_factor = (patch_size * merge_size) as f64;

    let mut h_bar = ((h / grid_factor).round() as usize).max(1) * patch_size * merge_size;
    let mut w_bar = ((w / grid_factor).round() as usize).max(1) * patch_size * merge_size;

    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = ((h / beta / grid_factor).floor() as usize).max(1) * patch_size * merge_size;
        w_bar = ((w / beta / grid_factor).floor() as usize).max(1) * patch_size * merge_size;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / grid_factor).ceil() as usize * patch_size * merge_size;
        w_bar = (w * beta / grid_factor).ceil() as usize * patch_size * merge_size;
    }

    Ok((h_bar, w_bar))
}

/// Normalize pixel values: `(pixel / 255.0 - mean) / std`.
///
/// Operates in-place on a flat `[C, H, W]` buffer (C=3 channels interleaved as CHW).
pub fn normalize_chw(pixels: &mut [f32], height: usize, width: usize, mean: [f32; 3], std: [f32; 3]) {
    let hw = height * width;
    for c in 0..3 {
        let offset = c * hw;
        let m = mean[c];
        let s = std[c];
        let inv_s = 1.0 / s;
        for i in 0..hw {
            pixels[offset + i] = (pixels[offset + i] - m) * inv_s;
        }
    }
}

/// Patchify an image into vision encoder input patches.
///
/// Input: normalized CHW float buffer of shape `[C, resized_h, resized_w]`.
/// The image is treated as a single frame (temporal=1), which gets duplicated
/// to meet `temporal_patch_size` (typically 2).
///
/// Output: flat `Vec<f32>` of shape `[grid_t * grid_h * grid_w, patch_dim]` where
/// `patch_dim = C * temporal_patch_size * patch_size * patch_size`.
///
/// Also returns `(grid_t, grid_h, grid_w)`.
pub fn patchify(
    chw_pixels: &[f32],
    resized_h: usize,
    resized_w: usize,
    config: &VisionConfig,
) -> Result<(Vec<f32>, usize, usize, usize)> {
    let c = config.in_channels;
    let ps = config.patch_size;
    let tp = config.temporal_patch_size;
    let ms = config.spatial_merge_size;

    if chw_pixels.len() != c * resized_h * resized_w {
        return Err(HerbertError::Config(format!(
            "patchify: expected {} elements ({}x{}x{}), got {}",
            c * resized_h * resized_w,
            c, resized_h, resized_w,
            chw_pixels.len()
        )));
    }

    let grid_t = 1usize; // single image → 1 temporal frame
    let grid_h = resized_h / ps;
    let grid_w = resized_w / ps;
    let patch_dim = c * tp * ps * ps;
    let num_patches = grid_t * grid_h * grid_w;

    // We have 1 frame but need `tp` frames. Duplicate the single frame.
    // Frames: frame[0] = frame[1] = chw_pixels.
    // Target layout after 10D reshape + permute (matching HF):
    //   (batch=1, grid_t, tp, C, grid_h//ms, ms, ps, grid_w//ms, ms, ps)
    //   → permute to (batch, grid_t, grid_h//ms, grid_w//ms, ms, ms, C, tp, ps, ps)
    //   → reshape to (num_patches, patch_dim)
    //
    // For a single image (grid_t=1, tp=2, frames duplicated), we produce
    // patches in the order matching HF's view+permute.

    let merged_h = grid_h / ms;
    let merged_w = grid_w / ms;

    let mut patches = vec![0.0f32; num_patches * patch_dim];

    // Iterate in the permuted order
    for bh in 0..merged_h {
        for bw in 0..merged_w {
            for mh in 0..ms {
                for mw in 0..ms {
                    // Patch index in output
                    let patch_idx = (bh * merged_w + bw) * (ms * ms) + mh * ms + mw;
                    let out_offset = patch_idx * patch_dim;

                    // Pixel row/col in the resized image
                    let row_start = (bh * ms + mh) * ps;
                    let col_start = (bw * ms + mw) * ps;

                    // Write patch_dim elements: [C, tp, ps, ps]
                    let mut idx = 0;
                    for ch in 0..c {
                        for _t in 0..tp {
                            // Both temporal frames are the same single image
                            for pr in 0..ps {
                                for pc in 0..ps {
                                    let r = row_start + pr;
                                    let col = col_start + pc;
                                    patches[out_offset + idx] =
                                        chw_pixels[ch * resized_h * resized_w + r * resized_w + col];
                                    idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((patches, grid_t, grid_h, grid_w))
}

/// Load an image from raw RGB bytes (H*W*3, row-major) and produce vision encoder input.
///
/// Returns `(patches, grid_t, grid_h, grid_w)`.
pub fn preprocess_rgb(
    rgb_bytes: &[u8],
    height: usize,
    width: usize,
    config: &VisionConfig,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(Vec<f32>, usize, usize, usize)> {
    if rgb_bytes.len() != height * width * 3 {
        return Err(HerbertError::Config(format!(
            "expected {} RGB bytes ({}x{}x3), got {}",
            height * width * 3,
            height, width,
            rgb_bytes.len()
        )));
    }

    // 1. Smart resize dimensions
    let (resized_h, resized_w) = smart_resize(
        height, width,
        config.patch_size,
        config.spatial_merge_size,
        min_pixels,
        max_pixels,
    )?;

    // 2. Resize using bilinear interpolation (simple implementation)
    let mut resized = vec![0u8; resized_h * resized_w * 3];
    bilinear_resize_rgb(rgb_bytes, height, width, &mut resized, resized_h, resized_w);

    // 3. Convert to CHW float [0, 1]
    let mut chw = vec![0.0f32; 3 * resized_h * resized_w];
    let hw = resized_h * resized_w;
    for y in 0..resized_h {
        for x in 0..resized_w {
            let src = (y * resized_w + x) * 3;
            for c in 0..3 {
                chw[c * hw + y * resized_w + x] = resized[src + c] as f32 / 255.0;
            }
        }
    }

    // 4. Normalize
    normalize_chw(&mut chw, resized_h, resized_w, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);

    // 5. Patchify
    patchify(&chw, resized_h, resized_w, config)
}

/// Simple bilinear resize for RGB images (HWC layout).
pub fn bilinear_resize_rgb(
    src: &[u8], src_h: usize, src_w: usize,
    dst: &mut [u8], dst_h: usize, dst_w: usize,
) {
    if dst_h == 0 || dst_w == 0 {
        return;
    }
    let scale_h = src_h as f64 / dst_h as f64;
    let scale_w = src_w as f64 / dst_w as f64;

    for dy in 0..dst_h {
        let sy = dy as f64 * scale_h;
        let y0 = (sy.floor() as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = sy - y0 as f64;

        for dx in 0..dst_w {
            let sx = dx as f64 * scale_w;
            let x0 = (sx.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = sx - x0 as f64;

            let dst_idx = (dy * dst_w + dx) * 3;
            for c in 0..3 {
                let v00 = src[(y0 * src_w + x0) * 3 + c] as f64;
                let v01 = src[(y0 * src_w + x1) * 3 + c] as f64;
                let v10 = src[(y1 * src_w + x0) * 3 + c] as f64;
                let v11 = src[(y1 * src_w + x1) * 3 + c] as f64;
                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                dst[dst_idx + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

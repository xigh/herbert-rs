//! 3D position IDs for MRoPE (Multimodal Rotary Position Embedding).
//!
//! Each token has a `Position3D { t, h, w }`:
//! - Text tokens: `(pos, pos, pos)` — sequential in all 3 dimensions.
//! - Image tokens: `(0, row, col)` — spatial grid coordinates.

/// 3D position for MRoPE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position3D {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl Position3D {
    /// Create a text position (same coordinate in all dimensions).
    #[inline]
    pub fn text(pos: u32) -> Self {
        Self { t: pos, h: pos, w: pos }
    }

    /// Create an image position with spatial grid coordinates.
    #[inline]
    pub fn image(row: u32, col: u32) -> Self {
        Self { t: 0, h: row, w: col }
    }

    /// Convert to `[t, h, w]` array for MRoPE cos/sin computation.
    #[inline]
    pub fn as_array(&self) -> [u32; 3] {
        [self.t, self.h, self.w]
    }
}

/// Information about an image embedded in a token sequence.
#[derive(Clone, Debug)]
pub struct ImageInfo {
    /// Token index where image tokens start.
    pub start_idx: usize,
    /// Height of image grid (number of rows of visual tokens).
    pub grid_h: usize,
    /// Width of image grid (number of columns of visual tokens).
    pub grid_w: usize,
}

impl ImageInfo {
    /// Number of visual tokens for this image.
    pub fn num_tokens(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Token index where image tokens end (exclusive).
    pub fn end_idx(&self) -> usize {
        self.start_idx + self.num_tokens()
    }
}

/// Build 3D position IDs for a VL sequence with mixed text and image tokens.
///
/// - Text tokens get `Position3D::text(text_pos++)` (sequential).
/// - Image tokens get `Position3D::image(row, col)` (spatial grid).
/// - Text after an image continues from the last text position.
pub fn build_vl_position_ids(
    total_len: usize,
    images: &[ImageInfo],
) -> Vec<Position3D> {
    let mut positions = Vec::with_capacity(total_len);
    let mut text_pos: u32 = 0;
    let mut token_idx: usize = 0;
    let mut img_idx: usize = 0;

    while token_idx < total_len {
        // Check if current token is the start of an image region
        if img_idx < images.len() && token_idx == images[img_idx].start_idx {
            let img = &images[img_idx];
            for row in 0..img.grid_h {
                for col in 0..img.grid_w {
                    positions.push(Position3D::image(row as u32, col as u32));
                }
            }
            token_idx += img.num_tokens();
            img_idx += 1;
        } else {
            // Text token
            positions.push(Position3D::text(text_pos));
            text_pos += 1;
            token_idx += 1;
        }
    }

    positions
}

/// Build 3D position IDs for a text-only sequence on a VL model.
///
/// Each token gets `Position3D::text(i)` for i = 0..seq_len.
pub fn build_text_position_ids(seq_len: usize) -> Vec<Position3D> {
    (0..seq_len).map(|i| Position3D::text(i as u32)).collect()
}

/// Assemble combined embeddings for a VL sequence with mixed text and image tokens.
///
/// - Text tokens get their embedding from `embed_tokens` (BF16, converted to f32 on lookup).
/// - Image regions get their embeddings from the corresponding `VisionEmbedding`.
///
/// Returns a flat `Vec<f32>` of length `total_len * hidden_size`.
pub fn build_vl_embeddings(
    tokens: &[u32],
    images: &[ImageInfo],
    image_hidden_states: &[&[f32]], // one slice per image, length = image.num_tokens * hidden_size
    embed_tokens: &[herbert_core::tensor::BF16],
    hidden_size: usize,
) -> Vec<f32> {
    let total_len = tokens.len();
    let mut out = vec![0.0f32; total_len * hidden_size];
    let mut token_idx = 0usize;
    let mut img_idx = 0usize;

    while token_idx < total_len {
        if img_idx < images.len() && token_idx == images[img_idx].start_idx {
            let img = &images[img_idx];
            let img_hs = image_hidden_states[img_idx];
            let n = img.num_tokens();
            let dst_start = token_idx * hidden_size;
            let copy_len = n * hidden_size;
            out[dst_start..dst_start + copy_len].copy_from_slice(&img_hs[..copy_len]);
            token_idx += n;
            img_idx += 1;
        } else {
            let t = tokens[token_idx] as usize;
            let src_start = t * hidden_size;
            let dst_start = token_idx * hidden_size;
            let src = &embed_tokens[src_start..src_start + hidden_size];
            let dst = &mut out[dst_start..dst_start + hidden_size];
            for i in 0..hidden_size {
                dst[i] = herbert_core::tensor::bf16_to_f32(src[i]);
            }
            token_idx += 1;
        }
    }

    out
}

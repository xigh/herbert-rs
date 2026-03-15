//! Q4 snapshot serialization helpers.

use crate::kernels::KernelWeight;
use crate::weight::{BF16Weight, INT8Weight, Q4Weight};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const SNAPSHOT_MAGIC: &[u8; 4] = b"HWS1";
const TAG_Q4: u8 = 0;
const TAG_INT8: u8 = 1;
const TAG_BF16: u8 = 2;

pub use herbert_backend_common::snapshot::{
    begin_decode_step, dump_bf16, dump_f32, dump_u32, dump_usize, dump_weight, end_decode_step,
    is_snapshot_active, load_bf16_vec, load_f32_vec, load_u32, load_usize, should_snapshot_layer,
};

pub fn dump_kernel_weight(w: &KernelWeight, path: &Path) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(SNAPSHOT_MAGIC)?;
    match w {
        KernelWeight::Q4(q) => {
            out.write_all(&[TAG_Q4])?;
            q.cache_write_inner(&mut out)?;
        }
        KernelWeight::INT8(i) => {
            out.write_all(&[TAG_INT8])?;
            i.cache_write_inner(&mut out)?;
        }
        KernelWeight::BF16(b) => {
            out.write_all(&[TAG_BF16])?;
            b.cache_write_inner(&mut out)?;
        }
    }
    out.flush()
}

pub fn load_kernel_weight(path: &Path) -> io::Result<KernelWeight> {
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    input.read_exact(&mut magic)?;
    if &magic != SNAPSHOT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} has invalid snapshot magic", path.display()),
        ));
    }

    let mut tag = [0u8; 1];
    input.read_exact(&mut tag)?;
    match tag[0] {
        TAG_Q4 => Ok(KernelWeight::Q4(Q4Weight::cache_read_inner(&mut input)?)),
        TAG_INT8 => Ok(KernelWeight::INT8(INT8Weight::cache_read_inner(&mut input)?)),
        TAG_BF16 => Ok(KernelWeight::BF16(BF16Weight::cache_read_inner(&mut input)?)),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} has unknown kernel tag {}", path.display(), other),
        )),
    }
}

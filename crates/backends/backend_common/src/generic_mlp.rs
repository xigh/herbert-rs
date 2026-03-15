//! Generic MLP module parameterized over LinearOps.

use std::marker::PhantomData;

use herbert_core::error::Result;

use crate::kernels as common_kernels;
use crate::linear_ops::LinearOps;

/// MLP activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpActivation {
    /// SwiGLU: silu(gate) * up (Qwen, Mistral, Llama, etc.)
    SwiGLU,
    /// GELU(tanh): gelu_tanh(gate) * up (Gemma3)
    GeluTanh,
}

/// MLP module generic over the linear ops / weight type.
pub struct GenericMLP<L: LinearOps> {
    pub gate_proj: L::Weight,
    pub up_proj: L::Weight,
    pub down_proj: L::Weight,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub activation: MlpActivation,
    pub _marker: PhantomData<L>,
}

impl<L: LinearOps> GenericMLP<L> {
    #[inline]
    fn apply_activation(&self, gate: &mut [f32], up: &[f32]) {
        match self.activation {
            MlpActivation::SwiGLU => L::swiglu(gate, up),
            MlpActivation::GeluTanh => common_kernels::gelu_tanh_inplace(gate, up),
        }
    }

    pub fn forward_decode(
        &self,
        x: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        debug_assert_eq!(gate.len(), self.intermediate_size);
        debug_assert_eq!(up.len(), self.intermediate_size);
        debug_assert_eq!(output.len(), self.hidden_size);
        // Try fused gate+up+swiglu path (activation applied during dequant epilogue).
        // Falls back to separate matvec + activation if backend doesn't support fusion.
        if self.activation == MlpActivation::SwiGLU {
            let fused = L::fused_gate_up_swiglu_matvec(
                x, &self.gate_proj, gate, &self.up_proj, up,
            )?;
            if fused {
                L::matvec(gate, &self.down_proj, output)?;
                return Ok(());
            }
        }
        L::fused_2_matvec(x, &self.gate_proj, gate, &self.up_proj, up)?;
        self.apply_activation(gate, up);
        L::matvec(gate, &self.down_proj, output)?;
        Ok(())
    }

    /// Single-threaded decode — uses matvec_st to bypass pool/autotuner overhead.
    pub fn forward_decode_st(
        &self,
        x: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        debug_assert_eq!(gate.len(), self.intermediate_size);
        debug_assert_eq!(up.len(), self.intermediate_size);
        debug_assert_eq!(output.len(), self.hidden_size);
        L::fused_gate_up_matvec_st(x, &self.gate_proj, &self.up_proj, gate, up)?;
        self.apply_activation(gate, up);
        L::matvec_st(gate, &self.down_proj, output)?;
        Ok(())
    }

    pub fn forward_prefill(
        &self,
        x: &[f32],
        seq_len: usize,
        gate: &mut [f32],
        up: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        L::matmul(x, &self.gate_proj, gate, seq_len)?;
        L::matmul(x, &self.up_proj, up, seq_len)?;
        self.apply_activation(gate, up);
        L::matmul(gate, &self.down_proj, output, seq_len)?;
        Ok(())
    }
}

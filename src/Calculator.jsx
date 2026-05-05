import React, { useState, useMemo } from 'react';

// ============================================================
// Generic FLOPs computation
// ============================================================
function computeFlopsPerToken(cfg) {
  const {
    hidden_size: d, num_hidden_layers: L, vocab_size: V, seq_len: T,
    causal_mfu = true,
    attention_type,
    num_attention_heads: nq, num_kv_heads: nkv, head_dim: dh,
    sliding_window, full_attn_every_n = 0,
    linear_num_key_heads: lnk, linear_num_value_heads: lnv,
    linear_key_head_dim: ldqk, linear_value_head_dim: ldv,
    linear_conv_kernel_dim: lck = 4,
    q_lora_rank, kv_lora_rank,
    qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
    ffn_type,
    intermediate_size: di,
    first_k_dense_replace = 0,
    moe_intermediate_size: dm,
    n_routed_experts, n_shared_experts, num_experts_per_tok,
    mtp_enabled = false, num_mtp_layers = 0,
  } = cfg;

  const cd = causal_mfu ? 2 : 1;

  // --- Attention ---
  let attn_lin = 0, attn_quad = 0;
  if (attention_type === 'mla') {
    const qhd = qk_nope_head_dim + qk_rope_head_dim;
    attn_lin = 2*d*q_lora_rank + 2*q_lora_rank*nq*qhd
             + 2*d*(kv_lora_rank + qk_rope_head_dim)
             + 2*kv_lora_rank*nq*(qk_nope_head_dim + v_head_dim)
             + 2*nq*v_head_dim*d;
    attn_quad = (2*nq*qhd*T + 2*nq*T*v_head_dim) / cd;
  } else if (attention_type === 'linear_hybrid') {
    // Hybrid: linear attention layers interleaved with full attention layers.
    // full_attn_every_n controls the period: 1 full per N layers (e.g., 4 = 3:1 linear:full).
    const full_attn_lin = 2*d*nq*dh + 2*d*nkv*dh + 2*d*nkv*dh + 2*nq*dh*d;
    const full_attn_quad = (2*nq*dh*T + 2*nq*T*dh) / cd;
    // Linear attention per-layer per-token (gated linear / Mamba-style approximation).
    // Q has same head count as V output, dim = key_head_dim.
    const lin_q_proj = 2*d*lnv*ldqk;
    const lin_k_proj = 2*d*lnk*ldqk;
    const lin_v_proj = 2*d*lnv*ldv;
    const lin_o_proj = 2*lnv*ldv*d;
    const lin_conv = 2*lck*(lnv*ldqk + lnk*ldqk);
    const lin_state = 2*lnk*ldqk*ldv;
    const lin_query = 2*lnv*ldqk*ldv;
    const lin_proj_total = lin_q_proj + lin_k_proj + lin_v_proj + lin_o_proj + lin_conv;
    const lin_attn_total = lin_state + lin_query;
    const n = full_attn_every_n && full_attn_every_n > 0 ? full_attn_every_n : 1;
    const full_share = 1 / n;
    const linear_share = 1 - full_share;
    attn_lin = full_share * full_attn_lin + linear_share * lin_proj_total;
    attn_quad = full_share * full_attn_quad + linear_share * lin_attn_total;
  } else {
    attn_lin = 2*d*nq*dh + 2*d*nkv*dh + 2*d*nkv*dh + 2*nq*dh*d;
    const attn_quad_full = (2*nq*dh*T + 2*nq*T*dh) / cd;
    if (attention_type === 'sliding' && sliding_window && sliding_window < T) {
      const attn_quad_sliding = 2*nq*dh*sliding_window + 2*nq*sliding_window*dh; // already directional
      if (full_attn_every_n && full_attn_every_n > 1) {
        const full_share = 1 / full_attn_every_n;
        attn_quad = full_share * attn_quad_full + (1 - full_share) * attn_quad_sliding;
      } else {
        attn_quad = attn_quad_sliding;
      }
    } else {
      attn_quad = attn_quad_full;
    }
  }
  const attn = attn_lin + attn_quad;

  // --- FFN ---
  const dense_ffn = 6 * d * di;
  let moe_ffn_total = 0, router = 0, moe_layer = 0;
  if (ffn_type === 'moe') {
    const expert = 6 * d * dm;
    moe_ffn_total = (num_experts_per_tok + n_shared_experts) * expert;
    router = 2 * d * n_routed_experts;
    moe_layer = attn + router + moe_ffn_total;
  }

  const dense_layer = attn + dense_ffn;
  const n_dense = ffn_type === 'moe' ? first_k_dense_replace : L;
  const n_moe = ffn_type === 'moe' ? L - first_k_dense_replace : 0;

  const lm_head = 2 * d * V;

  let mtp = 0;
  if (mtp_enabled && num_mtp_layers > 0) {
    const block = ffn_type === 'moe' ? moe_layer : dense_layer;
    const proj = 2 * (2*d) * d;
    mtp = num_mtp_layers * (block + proj + lm_head);
  }

  const fwd = n_dense*dense_layer + n_moe*moe_layer + mtp + lm_head;

  return {
    fwd_per_token: fwd,
    breakdown: {
      attn_linears: attn_lin,
      attn_quadratic: attn_quad,
      attn_total: attn,
      dense_ffn,
      moe_ffn_total,
      router,
      dense_layers_total: n_dense * dense_layer,
      moe_layers_total: n_moe * moe_layer,
      mtp_total: mtp,
      lm_head,
      attn_quadratic_share: attn_quad * L,
    },
  };
}

// ============================================================
// GPU Specs (per-GPU dense TFLOPS)
// ============================================================
const GPUS = {
  B300:  { fp4: 15000, fp8: 7000,  bf16: 3500, hbm: 288, name: 'B300 (Blackwell Ultra)' },
  B200:  { fp4: 9000,  fp8: 4500,  bf16: 2250, hbm: 192, name: 'B200 (Blackwell)' },
  H200:  { fp4: 0,     fp8: 1979,  bf16: 989,  hbm: 141, name: 'H200 (Hopper)' },
  H100:  { fp4: 0,     fp8: 1979,  bf16: 989,  hbm: 80,  name: 'H100 (Hopper)' },
  H800:  { fp4: 0,     fp8: 1979,  bf16: 989,  hbm: 80,  name: 'H800 (Hopper, China)' },
};

// ============================================================
// Model Presets
// ============================================================
const PRESETS = {
  'dsv3': {
    label: 'DeepSeek-V3 (671B / 37B-A)',
    cfg: {
      hidden_size: 7168, num_hidden_layers: 61, vocab_size: 129280, seq_len: 4096,
      attention_type: 'mla',
      num_attention_heads: 128,
      q_lora_rank: 1536, kv_lora_rank: 512,
      qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128,
      ffn_type: 'moe',
      intermediate_size: 18432, first_k_dense_replace: 3,
      moe_intermediate_size: 2048,
      n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 8,
      mtp_enabled: true, num_mtp_layers: 1,
    },
  },
  'qwen3-235b': {
    label: 'Qwen3-235B-A22B (MoE)',
    cfg: {
      hidden_size: 4096, num_hidden_layers: 94, vocab_size: 151936, seq_len: 4096,
      attention_type: 'gqa',
      num_attention_heads: 64, num_kv_heads: 4, head_dim: 128,
      ffn_type: 'moe',
      intermediate_size: 12288, first_k_dense_replace: 0,
      moe_intermediate_size: 1536,
      n_routed_experts: 128, n_shared_experts: 0, num_experts_per_tok: 8,
      mtp_enabled: false, num_mtp_layers: 0,
    },
  },
  'llama3-70b': {
    label: 'Llama 3 70B (Dense, GQA)',
    cfg: {
      hidden_size: 8192, num_hidden_layers: 80, vocab_size: 128256, seq_len: 8192,
      attention_type: 'gqa',
      num_attention_heads: 64, num_kv_heads: 8, head_dim: 128,
      ffn_type: 'dense', intermediate_size: 28672,
      mtp_enabled: false,
    },
  },
  'llama3-405b': {
    label: 'Llama 3 405B (Dense, GQA)',
    cfg: {
      hidden_size: 16384, num_hidden_layers: 126, vocab_size: 128256, seq_len: 8192,
      attention_type: 'gqa',
      num_attention_heads: 128, num_kv_heads: 8, head_dim: 128,
      ffn_type: 'dense', intermediate_size: 53248,
      mtp_enabled: false,
    },
  },
  'qwen3.5-397b': {
    label: 'Qwen3.5-397B-A17B (Linear+Full Hybrid MoE)',
    cfg: {
      hidden_size: 4096, num_hidden_layers: 60, vocab_size: 248320, seq_len: 4096,
      attention_type: 'linear_hybrid',
      num_attention_heads: 32, num_kv_heads: 2, head_dim: 256,
      full_attn_every_n: 4,
      linear_num_key_heads: 16, linear_num_value_heads: 64,
      linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
      ffn_type: 'moe',
      intermediate_size: 1024, first_k_dense_replace: 0,
      moe_intermediate_size: 1024,
      n_routed_experts: 512, n_shared_experts: 1, num_experts_per_tok: 10,
      mtp_enabled: true, num_mtp_layers: 1,
    },
  },
  'qwen3.5-35b': {
    label: 'Qwen3.5-35B-A3B (Linear+Full Hybrid MoE)',
    cfg: {
      hidden_size: 2048, num_hidden_layers: 40, vocab_size: 248320, seq_len: 4096,
      attention_type: 'linear_hybrid',
      num_attention_heads: 16, num_kv_heads: 2, head_dim: 256,
      full_attn_every_n: 4,
      linear_num_key_heads: 16, linear_num_value_heads: 32,
      linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
      ffn_type: 'moe',
      intermediate_size: 512, first_k_dense_replace: 0,
      moe_intermediate_size: 512,
      n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 8,
      mtp_enabled: true, num_mtp_layers: 1,
    },
  },
  'qwen2.5-72b': {
    label: 'Qwen2.5-72B (Dense, GQA)',
    cfg: {
      hidden_size: 8192, num_hidden_layers: 80, vocab_size: 152064, seq_len: 8192,
      attention_type: 'gqa',
      num_attention_heads: 64, num_kv_heads: 8, head_dim: 128,
      ffn_type: 'dense', intermediate_size: 29568,
      mtp_enabled: false,
    },
  },
};

// ============================================================
// Format helpers
// ============================================================
const fmt = (n, digits = 2) => {
  if (n === 0) return '0';
  const abs = Math.abs(n);
  // Plain notation for "human-readable" range: 0.01 to 999,999
  if (abs < 1e6 && abs >= 0.01) {
    // Use thousands separator for readability
    const fixed = n.toFixed(digits);
    const [int, frac] = fixed.split('.');
    const withSep = parseInt(int).toLocaleString('en-US');
    return frac ? `${withSep}.${frac}` : withSep;
  }
  // Scientific notation for very large or very small
  const exp = Math.floor(Math.log10(abs));
  const mantissa = n / Math.pow(10, exp);
  const supDigits = String(exp).split('').map(c => {
    if (c === '-') return '⁻';
    return '⁰¹²³⁴⁵⁶⁷⁸⁹'[parseInt(c)];
  }).join('');
  return `${mantissa.toFixed(digits)}×10${supDigits}`;
};
const fmtTime = (h) => {
  if (h < 1/60) return `${(h*3600).toFixed(0)} sec`;
  if (h < 1) return `${(h * 60).toFixed(1)} min`;
  if (h < 24) return `${h.toFixed(2)} hr`;
  return `${(h / 24).toFixed(2)} days`;
};

// ============================================================
// Main Component
// ============================================================
export default function Calculator() {
  // Mode: 'estimate' = predict wall-clock from MFU
  //       'measure'  = compute MFU from observed step time
  const [mode, setMode] = useState('estimate');

  // Model architecture
  const [arch, setArch] = useState({ ...PRESETS['dsv3'].cfg });

  // Training params (shared)
  const [numGpus, setNumGpus] = useState(4096);
  const [gpuType, setGpuType] = useState('B300');
  const [precision, setPrecision] = useState('bf16');
  const [recompute, setRecompute] = useState(false);

  // Estimate-mode-specific
  const [tokens, setTokens] = useState(30);
  const [mfu, setMfu] = useState(18);
  const [costPerHour, setCostPerHour] = useState(6.80);

  // Measure-mode-specific
  const [stepTime, setStepTime] = useState(1.32);   // sec
  const [gbsUnit, setGbsUnit] = useState('tokens'); // 'tokens' | 'samples'
  const [gbsTokens, setGbsTokens] = useState(4194304); // 4M
  const [gbsSamples, setGbsSamples] = useState(1024);

  // UI state
  const [activePreset, setActivePreset] = useState('dsv3');

  const applyPreset = (key) => {
    setArch({ ...PRESETS[key].cfg });
    setActivePreset(key);
  };

  const updateArch = (updates) => {
    setArch((a) => ({ ...a, ...updates }));
    setActivePreset('custom');
  };

  const result = useMemo(() => {
    const { fwd_per_token, breakdown } = computeFlopsPerToken(arch);
    const multiplier = recompute ? 4 : 3;
    const flops_per_token = fwd_per_token * multiplier;

    const gpu = GPUS[gpuType];
    const peak = gpu[precision] || gpu.fp8;
    const peak_flops_per_sec_per_gpu = peak * 1e12;
    const peak_flops_per_sec_cluster = peak_flops_per_sec_per_gpu * numGpus;

    if (mode === 'estimate') {
      // ---- Forward direction: MFU → wall-clock ----
      const totalTokens = tokens * 1e12;
      const total_flops = flops_per_token * totalTokens;
      const eff_flops_per_sec = peak_flops_per_sec_cluster * (mfu / 100);
      const seconds = total_flops / eff_flops_per_sec;
      const hours = seconds / 3600;
      const days = hours / 24;
      const gpu_hours = hours * numGpus;
      const cost = gpu_hours * costPerHour;
      const tps = totalTokens / seconds;
      const tps_per_gpu = tps / numGpus;
      return {
        mode: 'estimate',
        fwd_per_token, flops_per_token, total_flops, breakdown,
        multiplier, peak,
        eff_flops_per_sec,
        peak_flops_per_sec_cluster, peak_flops_per_sec_per_gpu,
        seconds, hours, days, gpu_hours, cost,
        tps, tps_per_gpu,
      };
    } else {
      // ---- Inverse direction: step time → MFU ----
      const gbs_tokens = gbsUnit === 'tokens' ? gbsTokens : gbsSamples * arch.seq_len;
      const step_flops = gbs_tokens * flops_per_token;
      const step_flops_per_gpu = step_flops / numGpus;
      const eff_flops_per_sec_per_gpu = step_flops_per_gpu / stepTime;
      const eff_flops_per_sec_cluster = eff_flops_per_sec_per_gpu * numGpus;
      const inferred_mfu = (eff_flops_per_sec_per_gpu / peak_flops_per_sec_per_gpu) * 100;
      const tps = gbs_tokens / stepTime;
      const tps_per_gpu = tps / numGpus;
      // Also report BF16-equivalent MFU for cross-comparison
      const bf16_peak = gpu.bf16 * 1e12;
      const inferred_mfu_bf16 = (eff_flops_per_sec_per_gpu / bf16_peak) * 100;
      return {
        mode: 'measure',
        fwd_per_token, flops_per_token, breakdown,
        multiplier, peak,
        gbs_tokens, step_flops,
        eff_flops_per_sec_per_gpu, eff_flops_per_sec_cluster,
        peak_flops_per_sec_cluster, peak_flops_per_sec_per_gpu,
        inferred_mfu, inferred_mfu_bf16,
        tps, tps_per_gpu,
      };
    }
  }, [
    mode, arch, tokens, numGpus, gpuType, precision, mfu, recompute, costPerHour,
    stepTime, gbsUnit, gbsTokens, gbsSamples,
  ]);

  return (
    <div style={S.root}>
      <div style={S.bgGrid} />
      <div style={S.container}>
        <Header />

        {/* Mode tabs */}
        <ModeTabs mode={mode} setMode={setMode} />

        {/* Preset selector */}
        <section style={S.presetBar}>
          <div style={S.presetLabel}>
            <span style={S.presetMark} />
            Preset
          </div>
          <div style={S.presetButtons}>
            {Object.entries(PRESETS).map(([k, p]) => (
              <button
                key={k}
                onClick={() => applyPreset(k)}
                style={{...S.presetBtn, ...(activePreset === k ? S.presetBtnActive : {})}}
              >
                {p.label}
              </button>
            ))}
            {activePreset === 'custom' && (
              <span style={S.customBadge}>● custom</span>
            )}
          </div>
        </section>

        {/* Two-column main grid */}
        <div style={S.mainGrid}>
          {/* LEFT: Architecture */}
          <Panel num="01" title="Model architecture">
            <FieldGroup title="Layers & dimensions">
              <Field label="Hidden size">
                <Num value={arch.hidden_size} onChange={(v) => updateArch({hidden_size: v})} step={128} />
              </Field>
              <Field label="Num layers">
                <Num value={arch.num_hidden_layers} onChange={(v) => updateArch({num_hidden_layers: v})} />
              </Field>
              <Field label="Vocab size">
                <Num value={arch.vocab_size} onChange={(v) => updateArch({vocab_size: v})} step={1024} />
              </Field>
              <Field label="Sequence length">
                <Num value={arch.seq_len} onChange={(v) => updateArch({seq_len: v})} step={1024} />
                <Chips
                  options={[2048, 4096, 8192, 16384, 32768]}
                  value={arch.seq_len}
                  onChange={(v) => updateArch({seq_len: v})}
                />
              </Field>
            </FieldGroup>

            <FieldGroup title="Attention type">
              <div style={S.segRow}>
                {[
                  {k: 'mha', label: 'MHA', hint: 'Multi-Head'},
                  {k: 'gqa', label: 'GQA', hint: 'Grouped Query'},
                  {k: 'mqa', label: 'MQA', hint: 'Multi-Query'},
                  {k: 'sliding', label: 'Sliding', hint: 'Sliding Window'},
                  {k: 'mla', label: 'MLA', hint: 'Multi-head Latent'},
                  {k: 'linear_hybrid', label: 'L-Hybrid', hint: 'Linear+Full'},
                ].map(({k, label, hint}) => (
                  <button
                    key={k}
                    onClick={() => {
                      const updates = {attention_type: k};
                      if (k === 'mla') {
                        if (arch.q_lora_rank == null) updates.q_lora_rank = 1536;
                        if (arch.kv_lora_rank == null) updates.kv_lora_rank = 512;
                        if (arch.qk_nope_head_dim == null) updates.qk_nope_head_dim = 128;
                        if (arch.qk_rope_head_dim == null) updates.qk_rope_head_dim = 64;
                        if (arch.v_head_dim == null) updates.v_head_dim = 128;
                      } else if (k === 'linear_hybrid') {
                        if (arch.num_kv_heads == null) updates.num_kv_heads = 2;
                        if (arch.head_dim == null) updates.head_dim = 256;
                        if (arch.full_attn_every_n == null || arch.full_attn_every_n < 2) updates.full_attn_every_n = 4;
                        if (arch.linear_num_key_heads == null) updates.linear_num_key_heads = 16;
                        if (arch.linear_num_value_heads == null) updates.linear_num_value_heads = 32;
                        if (arch.linear_key_head_dim == null) updates.linear_key_head_dim = 128;
                        if (arch.linear_value_head_dim == null) updates.linear_value_head_dim = 128;
                        if (arch.linear_conv_kernel_dim == null) updates.linear_conv_kernel_dim = 4;
                      } else {
                        if (k === 'mha') {
                          updates.num_kv_heads = arch.num_attention_heads;
                        } else if (k === 'mqa') {
                          updates.num_kv_heads = 1;
                        } else if (arch.num_kv_heads == null) {
                          updates.num_kv_heads = arch.num_attention_heads;
                        }
                        if (arch.head_dim == null) updates.head_dim = 128;
                        if (k === 'sliding' && (arch.sliding_window == null || arch.sliding_window >= arch.seq_len / 2)) {
                          updates.sliding_window = Math.max(512, Math.min(4096, Math.floor(arch.seq_len / 4)));
                        }
                        if (k === 'sliding' && arch.full_attn_every_n == null) {
                          updates.full_attn_every_n = 4;
                        }
                      }
                      updateArch(updates);
                    }}
                    style={{...S.seg, ...(arch.attention_type === k ? S.segActive : {})}}
                  >
                    <div style={S.segLabel}>{label}</div>
                    <div style={S.segHint}>{hint}</div>
                  </button>
                ))}
              </div>

              {arch.attention_type === 'mla' ? (
                <div style={S.subSection}>
                  <div style={S.subTitle}>MLA parameters</div>
                  <Field label="num_attention_heads">
                    <Num value={arch.num_attention_heads} onChange={(v) => updateArch({num_attention_heads: v})} />
                  </Field>
                  <Field label="q_lora_rank">
                    <Num value={arch.q_lora_rank} onChange={(v) => updateArch({q_lora_rank: v})} />
                  </Field>
                  <Field label="kv_lora_rank">
                    <Num value={arch.kv_lora_rank} onChange={(v) => updateArch({kv_lora_rank: v})} />
                  </Field>
                  <FieldRow>
                    <Field label="qk_nope_head_dim">
                      <Num value={arch.qk_nope_head_dim} onChange={(v) => updateArch({qk_nope_head_dim: v})} />
                    </Field>
                    <Field label="qk_rope_head_dim">
                      <Num value={arch.qk_rope_head_dim} onChange={(v) => updateArch({qk_rope_head_dim: v})} />
                    </Field>
                    <Field label="v_head_dim">
                      <Num value={arch.v_head_dim} onChange={(v) => updateArch({v_head_dim: v})} />
                    </Field>
                  </FieldRow>
                </div>
              ) : arch.attention_type === 'linear_hybrid' ? (
                <>
                  <div style={S.subSection}>
                    <div style={S.subTitle}>Full-attention layer parameters</div>
                    <FieldRow>
                      <Field label="num_attention_heads">
                        <Num value={arch.num_attention_heads} onChange={(v) => updateArch({num_attention_heads: v})} />
                      </Field>
                      <Field label="num_kv_heads">
                        <Num
                          value={arch.num_kv_heads ?? arch.num_attention_heads}
                          onChange={(v) => updateArch({num_kv_heads: v})}
                        />
                      </Field>
                      <Field label="head_dim">
                        <Num value={arch.head_dim ?? 128} onChange={(v) => updateArch({head_dim: v})} />
                      </Field>
                    </FieldRow>
                    <Field label="full_attn_every_n (1 full per N layers)">
                      <Num
                        value={arch.full_attn_every_n ?? 4}
                        onChange={(v) => updateArch({full_attn_every_n: v})}
                      />
                    </Field>
                  </div>
                  <div style={S.subSection}>
                    <div style={S.subTitle}>Linear-attention layer parameters</div>
                    <FieldRow>
                      <Field label="linear_num_key_heads">
                        <Num
                          value={arch.linear_num_key_heads ?? 16}
                          onChange={(v) => updateArch({linear_num_key_heads: v})}
                        />
                      </Field>
                      <Field label="linear_num_value_heads">
                        <Num
                          value={arch.linear_num_value_heads ?? 32}
                          onChange={(v) => updateArch({linear_num_value_heads: v})}
                        />
                      </Field>
                    </FieldRow>
                    <FieldRow>
                      <Field label="linear_key_head_dim">
                        <Num
                          value={arch.linear_key_head_dim ?? 128}
                          onChange={(v) => updateArch({linear_key_head_dim: v})}
                        />
                      </Field>
                      <Field label="linear_value_head_dim">
                        <Num
                          value={arch.linear_value_head_dim ?? 128}
                          onChange={(v) => updateArch({linear_value_head_dim: v})}
                        />
                      </Field>
                      <Field label="linear_conv_kernel_dim">
                        <Num
                          value={arch.linear_conv_kernel_dim ?? 4}
                          onChange={(v) => updateArch({linear_conv_kernel_dim: v})}
                        />
                      </Field>
                    </FieldRow>
                    <div style={S.note}>
                      Linear attention approximated as Q/K/V projections + depthwise conv +
                      state update (K^T·V outer product) + query application (Q·state) per token.
                      Linear FLOPs scale O(d²) per token vs full attention's O(N·d).
                    </div>
                  </div>
                </>
              ) : (
                <div style={S.subSection}>
                  <div style={S.subTitle}>Attention parameters</div>
                  <FieldRow>
                    <Field label="num_attention_heads">
                      <Num value={arch.num_attention_heads} onChange={(v) => updateArch({num_attention_heads: v})} />
                    </Field>
                    <Field label="num_kv_heads">
                      <Num
                        value={arch.num_kv_heads ?? arch.num_attention_heads}
                        onChange={(v) => updateArch({num_kv_heads: v})}
                      />
                    </Field>
                    <Field label="head_dim">
                      <Num value={arch.head_dim ?? 128} onChange={(v) => updateArch({head_dim: v})} />
                    </Field>
                  </FieldRow>
                  {arch.attention_type === 'sliding' && (
                    <FieldRow>
                      <Field label="sliding_window (tokens)">
                        <Num
                          value={arch.sliding_window ?? 4096}
                          onChange={(v) => updateArch({sliding_window: v})}
                          step={1024}
                        />
                      </Field>
                      <Field label="full_attn_every_n (0=pure)">
                        <Num
                          value={arch.full_attn_every_n ?? 4}
                          onChange={(v) => updateArch({full_attn_every_n: v})}
                        />
                      </Field>
                    </FieldRow>
                  )}
                  {arch.attention_type === 'sliding' && arch.full_attn_every_n > 1 && (
                    <div style={S.note}>
                      Hybrid: 1 full-attention layer every {arch.full_attn_every_n} layers
                      (i.e., {arch.full_attn_every_n - 1}:1 sliding:full ratio).
                      Set to 0 or 1 for pure sliding.
                    </div>
                  )}
                  {arch.attention_type === 'mqa' && (
                    <div style={S.note}>
                      MQA = GQA with num_kv_heads = 1. The KV head count above will be ignored if not set to 1.
                    </div>
                  )}
                </div>
              )}
            </FieldGroup>

            <FieldGroup title="Feed-forward network">
              <div style={S.segRow}>
                <button
                  onClick={() => updateArch({ffn_type: 'dense'})}
                  style={{...S.seg, ...(arch.ffn_type === 'dense' ? S.segActive : {})}}
                >
                  <div style={S.segLabel}>Dense</div>
                  <div style={S.segHint}>SwiGLU</div>
                </button>
                <button
                  onClick={() => {
                    const updates = {ffn_type: 'moe'};
                    if (arch.moe_intermediate_size == null) updates.moe_intermediate_size = 2048;
                    if (arch.n_routed_experts == null) updates.n_routed_experts = 256;
                    if (arch.n_shared_experts == null) updates.n_shared_experts = 1;
                    if (arch.num_experts_per_tok == null) updates.num_experts_per_tok = 8;
                    if (arch.first_k_dense_replace == null) updates.first_k_dense_replace = 0;
                    updateArch(updates);
                  }}
                  style={{...S.seg, ...(arch.ffn_type === 'moe' ? S.segActive : {})}}
                >
                  <div style={S.segLabel}>MoE</div>
                  <div style={S.segHint}>Sparse experts</div>
                </button>
              </div>

              <Field label={arch.ffn_type === 'moe' ? 'Dense FFN inter (early layers)' : 'FFN intermediate size'}>
                <Num value={arch.intermediate_size} onChange={(v) => updateArch({intermediate_size: v})} step={256} />
              </Field>

              {arch.ffn_type === 'moe' && (
                <div style={S.subSection}>
                  <div style={S.subTitle}>MoE parameters</div>
                  <FieldRow>
                    <Field label="first_k_dense_replace">
                      <Num value={arch.first_k_dense_replace} onChange={(v) => updateArch({first_k_dense_replace: v})} />
                    </Field>
                    <Field label="moe_intermediate_size">
                      <Num value={arch.moe_intermediate_size} onChange={(v) => updateArch({moe_intermediate_size: v})} step={128} />
                    </Field>
                  </FieldRow>
                  <FieldRow>
                    <Field label="n_routed_experts">
                      <Num value={arch.n_routed_experts} onChange={(v) => updateArch({n_routed_experts: v})} />
                    </Field>
                    <Field label="n_shared_experts">
                      <Num value={arch.n_shared_experts} onChange={(v) => updateArch({n_shared_experts: v})} />
                    </Field>
                    <Field label="num_experts_per_tok">
                      <Num value={arch.num_experts_per_tok} onChange={(v) => updateArch({num_experts_per_tok: v})} />
                    </Field>
                  </FieldRow>
                </div>
              )}
            </FieldGroup>

            <FieldGroup title="Multi-Token Prediction (MTP)">
              <Toggle
                label="Enable MTP"
                hint="Adds an extra transformer block + linear projection [2d→d] + LM head per MTP layer"
                value={arch.mtp_enabled}
                onChange={(v) => {
                  const updates = {mtp_enabled: v};
                  if (v && arch.num_mtp_layers == null) updates.num_mtp_layers = 1;
                  updateArch(updates);
                }}
              />
              {arch.mtp_enabled && (
                <Field label="Number of MTP layers">
                  <Num value={arch.num_mtp_layers ?? 1} onChange={(v) => updateArch({num_mtp_layers: v})} />
                </Field>
              )}
            </FieldGroup>
          </Panel>

          {/* RIGHT: Training & Results */}
          <div style={S.rightCol}>
            <Panel num="02" title={mode === 'estimate' ? 'Training setup' : 'Observed measurement'}>
              <FieldRow>
                <Field label="Number of GPUs">
                  <Num value={numGpus} onChange={setNumGpus} step={8} />
                </Field>
                {mode === 'estimate' ? (
                  <Field label="Training tokens (T)">
                    <Num value={tokens} onChange={setTokens} step={0.1} />
                  </Field>
                ) : (
                  <Field label="Step time (sec)">
                    <Num value={stepTime} onChange={setStepTime} step={0.01} />
                  </Field>
                )}
              </FieldRow>

              <Field label="GPU model">
                <div style={S.segRow}>
                  {Object.entries(GPUS).map(([k, g]) => (
                    <button
                      key={k}
                      onClick={() => setGpuType(k)}
                      style={{...S.gpuSeg, ...(gpuType === k ? S.gpuSegActive : {})}}
                    >
                      <div style={S.segLabel}>{k}</div>
                      <div style={S.segHint}>{g.hbm}GB</div>
                    </button>
                  ))}
                </div>
              </Field>

              <Field label={`Precision (${GPUS[gpuType].name} dense peak)`}>
                <div style={S.segRow}>
                  {['bf16', 'fp8', 'fp4'].map(p => {
                    const v = GPUS[gpuType][p];
                    const disabled = !v;
                    return (
                      <button
                        key={p}
                        disabled={disabled}
                        onClick={() => !disabled && setPrecision(p)}
                        style={{
                          ...S.precSeg,
                          ...(precision === p ? S.precSegActive : {}),
                          ...(disabled ? S.precSegDisabled : {}),
                        }}
                      >
                        <div style={S.segLabel}>{p.toUpperCase()}</div>
                        <div style={S.segHint}>
                          {disabled ? '—' : `${v.toLocaleString()} TF`}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </Field>

              {mode === 'estimate' ? (
                <Field label="MFU (%)">
                  <Num value={mfu} onChange={setMfu} step={1} />
                  <Slider value={mfu} onChange={setMfu} min={5} max={70} step={1} />
                  <MfuRefBox gpuType={gpuType} precision={precision} />
                </Field>
              ) : (
                <Field label="Global batch size">
                  <div style={S.segRow}>
                    <button
                      onClick={() => setGbsUnit('tokens')}
                      style={{...S.seg, ...(gbsUnit === 'tokens' ? S.segActive : {})}}
                    >
                      <div style={S.segLabel}>Tokens</div>
                      <div style={S.segHint}>direct</div>
                    </button>
                    <button
                      onClick={() => setGbsUnit('samples')}
                      style={{...S.seg, ...(gbsUnit === 'samples' ? S.segActive : {})}}
                    >
                      <div style={S.segLabel}>Samples</div>
                      <div style={S.segHint}>× seq_len</div>
                    </button>
                  </div>
                  {gbsUnit === 'tokens' ? (
                    <>
                      <Num value={gbsTokens} onChange={setGbsTokens} step={1024} />
                      <div style={S.note}>
                        e.g. 4,194,304 = 4M tokens (typical large-scale pretraining)
                      </div>
                    </>
                  ) : (
                    <>
                      <Num value={gbsSamples} onChange={setGbsSamples} step={8} />
                      <div style={S.note}>
                        Effective tokens = {gbsSamples} × {arch.seq_len.toLocaleString()} = {(gbsSamples * arch.seq_len).toLocaleString()}
                      </div>
                    </>
                  )}
                </Field>
              )}

              <FieldRow>
                <Field label="Causal MFU">
                  <Toggle
                    value={arch.causal_mfu !== false}
                    onChange={(v) => updateArch({causal_mfu: v})}
                    inline
                    hint="Halves attention quadratic"
                  />
                </Field>
                <Field label="Activation recompute">
                  <Toggle
                    value={recompute}
                    onChange={setRecompute}
                    inline
                    hint="3× → 4× forward"
                  />
                </Field>
              </FieldRow>

              {mode === 'estimate' && (
                <Field label="GPU $/hour">
                  <Num value={costPerHour} onChange={setCostPerHour} step={0.1} />
                </Field>
              )}
            </Panel>

            {mode === 'estimate' ? (
              <Panel num="03" title="Estimated training time">
                <BigStat
                  label="WALL-CLOCK"
                  value={fmtTime(result.hours)}
                  sub={`${Math.round(result.gpu_hours).toLocaleString()} GPU-hr · $${(result.cost / 1e6).toFixed(2)}M`}
                />

                <div style={S.statGrid}>
                  <Stat label="Total FLOPs" value={fmt(result.total_flops, 2)} />
                  <Stat label="FLOPs / token" value={fmt(result.flops_per_token, 2)} />
                  <Stat label="Effective TFLOPS" value={fmt(result.eff_flops_per_sec / 1e12, 1)} />
                  <Stat label="Peak TFLOPS" value={fmt(result.peak_flops_per_sec_cluster / 1e12, 1)} />
                  <Stat label="Tokens/sec/GPU" value={fmt(result.tps_per_gpu, 0)} />
                  <Stat label="Days" value={result.days.toFixed(2)} />
                </div>

                <Breakdown breakdown={result.breakdown} fwd={result.fwd_per_token} arch={arch} />
              </Panel>
            ) : (
              <Panel num="03" title="Inferred MFU">
                <BigStat
                  label="MFU"
                  value={result.inferred_mfu.toFixed(2) + '%'}
                  sub={
                    <>
                      vs {precision.toUpperCase()} peak ({GPUS[gpuType][precision].toLocaleString()} TF/GPU)
                      {precision !== 'bf16' && (
                        <>  ·  BF16-equiv: {result.inferred_mfu_bf16.toFixed(2)}%</>
                      )}
                    </>
                  }
                />

                <div style={S.statGrid}>
                  <Stat
                    label="Achieved TFLOPS/GPU"
                    value={result.eff_flops_per_sec_per_gpu / 1e12 < 100
                      ? (result.eff_flops_per_sec_per_gpu / 1e12).toFixed(1)
                      : Math.round(result.eff_flops_per_sec_per_gpu / 1e12).toLocaleString()}
                  />
                  <Stat label="Cluster TFLOPS" value={fmt(result.eff_flops_per_sec_cluster / 1e12, 1)} />
                  <Stat label="GBS (tokens)" value={result.gbs_tokens.toLocaleString()} />
                  <Stat label="Step FLOPs" value={fmt(result.step_flops, 2)} />
                  <Stat label="FLOPs / token" value={fmt(result.flops_per_token, 2)} />
                  <Stat label="Tokens/sec/GPU" value={fmt(result.tps_per_gpu, 0)} />
                </div>

                <MfuComparison inferred={result.inferred_mfu} gpuType={gpuType} precision={precision} />

                <Breakdown breakdown={result.breakdown} fwd={result.fwd_per_token} arch={arch} />
              </Panel>
            )}
          </div>
        </div>

        <Methodology />
        <Footer />
      </div>
    </div>
  );
}

// ============================================================
// Sub-components
// ============================================================
function Header() {
  return (
    <header style={S.header}>
      <div style={S.tagRow}>
        <span style={S.tagAccent}>Vol. 2 · Issue 1</span>
        <span style={S.tagDivider}>—</span>
        <span style={S.tagDim}>An LLM training-time &amp; MFU calculator</span>
      </div>
      <h1 style={S.title}>
        Training&#8202;Time
        <span style={S.titleAccent}>Estimator</span>
      </h1>
      <p style={S.subtitle}>
        A first-order calculator for transformer pre-training wall-clock — or, in the
        inverse direction, for the MFU implied by a measured step time. Supports
        <em> MHA, GQA, MQA, MLA, Sliding-window</em> and <em>Linear+Full hybrid</em> attention,
        dense and mixture-of-experts FFN, and optional multi-token prediction.
      </p>
    </header>
  );
}

function ModeTabs({ mode, setMode }) {
  const tabs = [
    {
      k: 'estimate',
      title: 'Estimate training time',
      hint: 'MFU → wall-clock',
      icon: '→',
    },
    {
      k: 'measure',
      title: 'Measure achieved MFU',
      hint: 'step time → MFU',
      icon: '←',
    },
  ];
  return (
    <div style={S.modeTabs}>
      {tabs.map(t => (
        <button
          key={t.k}
          onClick={() => setMode(t.k)}
          style={{...S.modeTab, ...(mode === t.k ? S.modeTabActive : {})}}
        >
          <span style={S.modeTabIcon}>{t.icon}</span>
          <span style={S.modeTabBody}>
            <span style={S.modeTabTitle}>{t.title}</span>
            <span style={S.modeTabHint}>{t.hint}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

function MfuComparison({ inferred, gpuType, precision }) {
  // Reference MFU values from real-world benchmarks (NVIDIA Megatron-Core, 2026/03)
  const refs = [
    { label: 'DSV3 / GB200 / FP8 (MC)',     val: 23.3, gpu: 'B200', prec: 'fp8' },
    { label: 'DSV3 / GB300 / FP8 (MC)',     val: 17.6, gpu: 'B300', prec: 'fp8' },
  ].sort((a, b) => Math.abs(a.val - inferred) - Math.abs(b.val - inferred));

  return (
    <div style={S.cmpBox}>
      <div style={S.cmpTitle}>How does this compare?</div>
      <div style={S.cmpList}>
        {refs.map((r, i) => {
          const diff = inferred - r.val;
          const isMatch = r.gpu === gpuType && r.prec === precision;
          return (
            <div key={i} style={{...S.cmpRow, ...(isMatch ? S.cmpRowMatch : {})}}>
              <span style={S.cmpLabel}>{r.label}</span>
              <span style={S.cmpVal}>{r.val.toFixed(1)}%</span>
              <span style={{
                ...S.cmpDiff,
                color: diff >= 0 ? C.good : C.warn,
              }}>
                {diff >= 0 ? '+' : ''}{diff.toFixed(1)}pp
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Panel({ num, title, children }) {
  return (
    <section style={S.panel}>
      <div style={S.panelHead}>
        <span style={S.panelNum}>{num}</span>
        <h2 style={S.panelTitle}>{title}</h2>
      </div>
      <div style={S.panelBody}>{children}</div>
    </section>
  );
}

function FieldGroup({ title, children }) {
  return (
    <div style={S.fieldGroup}>
      <div style={S.fieldGroupTitle}>{title}</div>
      {children}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={S.field}>
      <label style={S.fieldLabel}>{label}</label>
      <div style={S.fieldBody}>{children}</div>
    </div>
  );
}

function FieldRow({ children }) {
  return <div style={S.fieldRow}>{children}</div>;
}

function Num({ value, onChange, step = 1 }) {
  return (
    <input
      type="number"
      value={value ?? ''}
      step={step}
      onChange={(e) => {
        const v = e.target.value === '' ? 0 : parseFloat(e.target.value);
        if (!isNaN(v)) onChange(v);
      }}
      style={S.input}
    />
  );
}

function Slider({ value, onChange, min, max, step }) {
  return (
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      style={S.slider}
    />
  );
}

function Chips({ options, value, onChange }) {
  return (
    <div style={S.chipRow}>
      {options.map(o => (
        <button
          key={o}
          onClick={() => onChange(o)}
          style={{...S.chip, ...(value === o ? S.chipActive : {})}}
        >{o.toLocaleString()}</button>
      ))}
    </div>
  );
}

function Toggle({ label, hint, value, onChange, inline }) {
  return (
    <label style={inline ? S.toggleInline : S.toggleRow}>
      <button
        type="button"
        onClick={() => onChange(!value)}
        style={{...S.toggleBtn, ...(value ? S.toggleBtnOn : {})}}
        aria-checked={value}
        role="switch"
      >
        <span style={{...S.toggleKnob, ...(value ? S.toggleKnobOn : {})}} />
      </button>
      {(label || hint) && (
        <div style={S.toggleText}>
          {label && <div style={S.toggleLabel}>{label}</div>}
          {hint && <div style={S.toggleHint}>{hint}</div>}
        </div>
      )}
    </label>
  );
}

function MfuRefBox({ gpuType, precision }) {
  const refs = [
    { label: 'DSV3 / B300 / FP8 (NVIDIA MC)', val: '~18%', match: gpuType === 'B300' && precision === 'fp8' },
    { label: 'DSV3 / B200 / FP8 (NVIDIA MC)', val: '~23%', match: gpuType === 'B200' && precision === 'fp8' },
  ];
  return (
    <div style={S.mfuRefBox}>
      <div style={S.mfuRefTitle}>Reference points</div>
      <div style={S.mfuRefList}>
        {refs.map((r, i) => (
          <div key={i} style={{...S.mfuRefItem, ...(r.match ? S.mfuRefItemMatch : {})}}>
            <span style={S.mfuRefLabel}>{r.label}</span>
            <span style={S.mfuRefVal}>{r.val}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BigStat({ label, value, sub }) {
  return (
    <div style={S.bigStat}>
      <div style={S.bigStatLabel}>{label}</div>
      <div style={S.bigStatValue}>{value}</div>
      <div style={S.bigStatSub}>{sub}</div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div style={S.stat}>
      <div style={S.statLabel}>{label}</div>
      <div style={S.statValue}>{value}</div>
    </div>
  );
}

// ============================================================
// Per-component formula builder (symbolic + substituted)
// ============================================================
function buildFormulas(a) {
  const d = a.hidden_size;
  const T = a.seq_len;
  const cd = a.causal_mfu !== false ? 2 : 1;
  const cdLabel = a.causal_mfu !== false ? ' / 2' : '';
  const formulas = [];

  // ---- Attention linears + quadratic ----
  if (a.attention_type === 'linear_hybrid') {
    const nq = a.num_attention_heads;
    const nkv = a.num_kv_heads ?? nq;
    const dh = a.head_dim ?? 128;
    const lnk = a.linear_num_key_heads ?? 16;
    const lnv = a.linear_num_value_heads ?? 32;
    const ldqk = a.linear_key_head_dim ?? 128;
    const ldv = a.linear_value_head_dim ?? 128;
    const lck = a.linear_conv_kernel_dim ?? 4;
    const n = a.full_attn_every_n && a.full_attn_every_n > 0 ? a.full_attn_every_n : 1;
    const fs = 1 / n;
    const ss = 1 - fs;

    formulas.push({
      group: `Full-attention layer (1 per ${n} layers, weight=${fs.toFixed(3)})`,
      items: [
        { name: 'Q proj',    sym: '2·d·n_q·head_dim',  sub: `2·${d}·${nq}·${dh}`,  val: 2*d*nq*dh },
        { name: 'K proj',    sym: '2·d·n_kv·head_dim', sub: `2·${d}·${nkv}·${dh}`, val: 2*d*nkv*dh },
        { name: 'V proj',    sym: '2·d·n_kv·head_dim', sub: `2·${d}·${nkv}·${dh}`, val: 2*d*nkv*dh },
        { name: 'O proj',    sym: '2·n_q·head_dim·d',  sub: `2·${nq}·${dh}·${d}`,  val: 2*nq*dh*d },
        { name: 'QKᵀ',       sym: `2·n_q·head_dim·seq${cdLabel}`, sub: `2·${nq}·${dh}·${T}${cdLabel}`, val: (2*nq*dh*T)/cd },
        { name: 'AV',        sym: `2·n_q·seq·head_dim${cdLabel}`, sub: `2·${nq}·${T}·${dh}${cdLabel}`, val: (2*nq*T*dh)/cd },
      ],
    });

    formulas.push({
      group: `Linear-attention layer (${n - 1} per ${n} layers, weight=${ss.toFixed(3)})`,
      items: [
        { name: 'Q proj',    sym: '2·d·n_v·d_qk',  sub: `2·${d}·${lnv}·${ldqk}`, val: 2*d*lnv*ldqk },
        { name: 'K proj',    sym: '2·d·n_k·d_qk',  sub: `2·${d}·${lnk}·${ldqk}`, val: 2*d*lnk*ldqk },
        { name: 'V proj',    sym: '2·d·n_v·d_v',   sub: `2·${d}·${lnv}·${ldv}`,  val: 2*d*lnv*ldv },
        { name: 'O proj',    sym: '2·n_v·d_v·d',   sub: `2·${lnv}·${ldv}·${d}`,  val: 2*lnv*ldv*d },
        { name: 'Conv (Q,K)',sym: '2·k·(n_v+n_k)·d_qk', sub: `2·${lck}·${lnv + lnk}·${ldqk}`, val: 2*lck*(lnv*ldqk + lnk*ldqk) },
        { name: 'KᵀV state', sym: '2·n_k·d_qk·d_v', sub: `2·${lnk}·${ldqk}·${ldv}`, val: 2*lnk*ldqk*ldv },
        { name: 'Q·state',   sym: '2·n_v·d_qk·d_v', sub: `2·${lnv}·${ldqk}·${ldv}`, val: 2*lnv*ldqk*ldv },
      ],
    });
  } else if (a.attention_type === 'mla') {
    const qhd = a.qk_nope_head_dim + a.qk_rope_head_dim;
    const q_a = 2 * d * a.q_lora_rank;
    const q_b = 2 * a.q_lora_rank * a.num_attention_heads * qhd;
    const kv_a = 2 * d * (a.kv_lora_rank + a.qk_rope_head_dim);
    const kv_b = 2 * a.kv_lora_rank * a.num_attention_heads * (a.qk_nope_head_dim + a.v_head_dim);
    const o_proj = 2 * a.num_attention_heads * a.v_head_dim * d;
    formulas.push({
      group: 'MLA attention linears (per layer)',
      items: [
        { name: 'q_a_proj',  sym: '2·d·q_lora', sub: `2·${d}·${a.q_lora_rank}`, val: q_a },
        { name: 'q_b_proj',  sym: '2·q_lora·n_h·(qk_nope+qk_rope)', sub: `2·${a.q_lora_rank}·${a.num_attention_heads}·${qhd}`, val: q_b },
        { name: 'kv_a_proj', sym: '2·d·(kv_lora+qk_rope)', sub: `2·${d}·${a.kv_lora_rank + a.qk_rope_head_dim}`, val: kv_a },
        { name: 'kv_b_proj', sym: '2·kv_lora·n_h·(qk_nope+v_head)', sub: `2·${a.kv_lora_rank}·${a.num_attention_heads}·${a.qk_nope_head_dim + a.v_head_dim}`, val: kv_b },
        { name: 'o_proj',    sym: '2·n_h·v_head·d', sub: `2·${a.num_attention_heads}·${a.v_head_dim}·${d}`, val: o_proj },
      ],
    });
    const attn_qk = (2 * a.num_attention_heads * qhd * T) / cd;
    const attn_pv = (2 * a.num_attention_heads * T * a.v_head_dim) / cd;
    formulas.push({
      group: 'Attention quadratic (per layer)',
      items: [
        { name: 'QKᵀ matmul', sym: `2·n_h·(qk_nope+qk_rope)·seq${cdLabel}`, sub: `2·${a.num_attention_heads}·${qhd}·${T}${cdLabel}`, val: attn_qk },
        { name: 'AV matmul',  sym: `2·n_h·seq·v_head${cdLabel}`, sub: `2·${a.num_attention_heads}·${T}·${a.v_head_dim}${cdLabel}`, val: attn_pv },
      ],
    });
  } else {
    const nq = a.num_attention_heads;
    const nkv = a.num_kv_heads ?? nq;
    const dh = a.head_dim ?? 128;
    const q_proj = 2 * d * nq * dh;
    const k_proj = 2 * d * nkv * dh;
    const v_proj = 2 * d * nkv * dh;
    const o_proj = 2 * nq * dh * d;
    const attnLabel = a.attention_type.toUpperCase();
    formulas.push({
      group: `${attnLabel} attention linears (per layer)`,
      items: [
        { name: 'Q proj', sym: '2·d·n_q·head_dim',  sub: `2·${d}·${nq}·${dh}`,  val: q_proj },
        { name: 'K proj', sym: '2·d·n_kv·head_dim', sub: `2·${d}·${nkv}·${dh}`, val: k_proj },
        { name: 'V proj', sym: '2·d·n_kv·head_dim', sub: `2·${d}·${nkv}·${dh}`, val: v_proj },
        { name: 'O proj', sym: '2·n_q·head_dim·d',  sub: `2·${nq}·${dh}·${d}`,  val: o_proj },
      ],
    });
    let attn_qk, attn_pv, qkSym, qkSub, pvSym, pvSub, qGroup;
    if (a.attention_type === 'sliding' && a.sliding_window && a.sliding_window < T) {
      const w = a.sliding_window;
      const n = a.full_attn_every_n;
      const qk_sliding = 2 * nq * dh * w;
      const pv_sliding = 2 * nq * w * dh;
      if (n && n > 1) {
        const fs = 1 / n;
        const ss = 1 - fs;
        const qk_full = (2 * nq * dh * T) / cd;
        const pv_full = (2 * nq * T * dh) / cd;
        attn_qk = ss * qk_sliding + fs * qk_full;
        attn_pv = ss * pv_sliding + fs * pv_full;
        const ssStr = ss.toFixed(3), fsStr = fs.toFixed(3);
        qkSym = `${ssStr}·(2·n_q·head_dim·window) + ${fsStr}·(2·n_q·head_dim·seq${cdLabel})`;
        qkSub = `${ssStr}·(2·${nq}·${dh}·${w}) + ${fsStr}·(2·${nq}·${dh}·${T}${cdLabel})`;
        pvSym = `${ssStr}·(2·n_q·window·head_dim) + ${fsStr}·(2·n_q·seq·head_dim${cdLabel})`;
        pvSub = `${ssStr}·(2·${nq}·${w}·${dh}) + ${fsStr}·(2·${nq}·${T}·${dh}${cdLabel})`;
        qGroup = `Hybrid attention (window=${w}, 1 full per ${n} layers, per-layer avg)`;
      } else {
        attn_qk = qk_sliding;
        attn_pv = pv_sliding;
        qkSym = '2·n_q·head_dim·window';
        qkSub = `2·${nq}·${dh}·${w}`;
        pvSym = '2·n_q·window·head_dim';
        pvSub = `2·${nq}·${w}·${dh}`;
        qGroup = `Sliding window attention (window=${w}, per layer)`;
      }
    } else {
      attn_qk = (2 * nq * dh * T) / cd;
      attn_pv = (2 * nq * T * dh) / cd;
      qkSym = `2·n_q·head_dim·seq${cdLabel}`;
      qkSub = `2·${nq}·${dh}·${T}${cdLabel}`;
      pvSym = `2·n_q·seq·head_dim${cdLabel}`;
      pvSub = `2·${nq}·${T}·${dh}${cdLabel}`;
      qGroup = 'Attention quadratic (per layer)';
    }
    formulas.push({
      group: qGroup,
      items: [
        { name: 'QKᵀ matmul', sym: qkSym, sub: qkSub, val: attn_qk },
        { name: 'AV matmul',  sym: pvSym, sub: pvSub, val: attn_pv },
      ],
    });
  }

  // ---- FFN ----
  if (a.ffn_type === 'dense') {
    const ffn = 6 * d * a.intermediate_size;
    formulas.push({
      group: 'Dense FFN — SwiGLU (per layer)',
      items: [
        { name: 'gate + up + down', sym: '6·d·intermediate', sub: `6·${d}·${a.intermediate_size}`, val: ffn },
      ],
    });
  } else {
    if (a.first_k_dense_replace > 0) {
      const ffn = 6 * d * a.intermediate_size;
      formulas.push({
        group: `Early dense layers — SwiGLU (×${a.first_k_dense_replace})`,
        items: [
          { name: 'gate + up + down', sym: '6·d·intermediate', sub: `6·${d}·${a.intermediate_size}`, val: ffn },
        ],
      });
    }
    const expert = 6 * d * a.moe_intermediate_size;
    const activated = a.num_experts_per_tok + a.n_shared_experts;
    const moe_total = activated * expert;
    const router = 2 * d * a.n_routed_experts;
    formulas.push({
      group: 'MoE FFN (per layer)',
      items: [
        { name: 'Router gating',  sym: '2·d·n_routed', sub: `2·${d}·${a.n_routed_experts}`, val: router },
        { name: 'Active experts', sym: '(top_k + n_shared)·6·d·moe_inter', sub: `${activated}·6·${d}·${a.moe_intermediate_size}`, val: moe_total },
      ],
    });
  }

  // ---- LM head ----
  const lm_head = 2 * d * a.vocab_size;
  formulas.push({
    group: 'LM head (final)',
    items: [
      { name: 'Output projection', sym: '2·d·vocab_size', sub: `2·${d}·${a.vocab_size}`, val: lm_head },
    ],
  });

  // ---- MTP ----
  if (a.mtp_enabled && a.num_mtp_layers > 0) {
    const proj = 2 * (2 * d) * d;
    formulas.push({
      group: `MTP module (×${a.num_mtp_layers}) — adds per layer:`,
      items: [
        { name: '1 transformer block', sym: '(attention + FFN above)', sub: 'see above', val: 0 },
        { name: 'Linear [2d→d]', sym: '2·(2d)·d', sub: `2·(2·${d})·${d}`, val: proj },
        { name: '1 LM head',     sym: '2·d·vocab_size', sub: `2·${d}·${a.vocab_size}`, val: lm_head },
      ],
    });
  }

  return formulas;
}


function Breakdown({ breakdown: b, fwd, arch }) {
  const [showFormulas, setShowFormulas] = useState(true);
  const items = [];
  if (b.dense_layers_total) items.push({label: `${arch.ffn_type === 'moe' ? arch.first_k_dense_replace : arch.num_hidden_layers} dense layers`, val: b.dense_layers_total});
  if (b.moe_layers_total) items.push({label: `${arch.num_hidden_layers - arch.first_k_dense_replace} MoE layers`, val: b.moe_layers_total});
  if (b.mtp_total) items.push({label: `MTP module (×${arch.num_mtp_layers})`, val: b.mtp_total});
  items.push({label: 'LM head', val: b.lm_head});
  items.push({label: '— attn quadratic share (across all layers)', val: b.attn_quadratic_share, sub: true});

  const formulas = buildFormulas(arch);

  return (
    <div style={S.breakdown}>
      <div style={S.subTitle}>FLOPs / token (forward) — high-level</div>
      {items.map((it, i) => (
        <div key={i} style={{...S.brRow, ...(it.sub ? S.brRowSub : {})}}>
          <span style={S.brLabel}>{it.label}</span>
          <span style={S.brBarOuter}>
            <span style={{
              ...S.brBarFill,
              width: `${Math.min(100, (it.val / fwd) * 100)}%`,
              ...(it.sub ? S.brBarFillSub : {}),
            }} />
          </span>
          <span style={S.brVal}>
            <span>{fmt(it.val, 1)}</span>
            <span style={S.brPct}>{((it.val / fwd) * 100).toFixed(1)}%</span>
          </span>
        </div>
      ))}

      <div style={S.formulaSection}>
        <button
          onClick={() => setShowFormulas((v) => !v)}
          style={S.formulaToggle}
          aria-expanded={showFormulas}
        >
          <span style={{...S.formulaCaret, transform: showFormulas ? 'rotate(90deg)' : 'rotate(0deg)'}}>▶</span>
          <span style={S.subTitle}>Per-component formulas</span>
          <span style={S.formulaToggleHint}>{showFormulas ? 'click to collapse' : 'click to expand'}</span>
        </button>
        {showFormulas && (
          <>
            {formulas.map((g, gi) => (
              <div key={gi} style={S.formulaGroup}>
                <div style={S.formulaGroupTitle}>{g.group}</div>
                {g.items.map((it, ii) => (
                  <div key={ii} style={S.formulaRow}>
                    <span style={S.formulaName}>{it.name}</span>
                    <span style={S.formulaBody}>
                      <span style={S.formulaSym}>{it.sym}</span>
                      {it.val > 0 && (
                        <>
                          <span style={S.formulaEq}>=</span>
                          <span style={S.formulaSub}>{it.sub}</span>
                          <span style={S.formulaEq}>=</span>
                          <span style={S.formulaVal}>{fmt(it.val, 2)}</span>
                        </>
                      )}
                      {it.val === 0 && (
                        <span style={S.formulaNote}>{it.sub}</span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            ))}
            <div style={S.formulaLegend}>
              d = hidden_size · n_h = num_attention_heads · n_q / n_kv = query/kv heads · seq = sequence length
              {arch.causal_mfu !== false ? ' · "/2" = causal halving' : ''}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Methodology() {
  return (
    <section style={S.method}>
      <div style={S.panelHead}>
        <span style={S.panelNum}>04</span>
        <h2 style={S.panelTitle}>Methodology</h2>
      </div>
      <div style={S.methodBody}>
        <div style={S.methodGrid}>
          <div style={S.methodCol}>
            <h3 style={S.methodTitle}>FLOPs accounting</h3>
            <ul style={S.bullets}>
              <li>Per matmul of shape <code style={S.code}>(M, K) × (K, N)</code>: <code style={S.code}>2·M·K·N</code> flops.</li>
              <li>Backward ≈ 2× forward; therefore one training step ≈ <strong>3×</strong> forward (or <strong>4×</strong> with full activation recompute).</li>
              <li>Causal masking halves the attention quadratic, expressed as <code style={S.code}>/2</code>.</li>
              <li>SwiGLU FFN: three matmuls (gate / up / down) → <code style={S.code}>6·d·d_inter</code>.</li>
              <li>Mixture-of-experts counts only the activated experts: <code style={S.code}>(top_k + n_shared) · 6·d·d_moe</code>; router cost is <code style={S.code}>2·d·n_routed</code>.</li>
              <li>MLA: <code style={S.code}>kv_b</code> projection uses <code style={S.code}>(qk_nope + v_head)</code>, excluding RoPE.</li>
              <li>Multi-token prediction (MTP): one extra transformer block, a <code style={S.code}>[2d → d]</code> projection, and an additional LM head per MTP layer.</li>
            </ul>
          </div>

          <div style={S.methodCol}>
            <h3 style={S.methodTitle}>Supported attention variants</h3>
            <dl style={S.dl}>
              <dt style={S.dt}>MHA</dt>
              <dd style={S.dd}>Multi-head attention; <code style={S.code}>n_kv = n_q</code>.</dd>
              <dt style={S.dt}>GQA</dt>
              <dd style={S.dd}>Grouped-query attention; <code style={S.code}>1 &lt; n_kv &lt; n_q</code>.</dd>
              <dt style={S.dt}>MQA</dt>
              <dd style={S.dd}>Multi-query attention; <code style={S.code}>n_kv = 1</code>.</dd>
              <dt style={S.dt}>MLA</dt>
              <dd style={S.dd}>Multi-head latent attention (DeepSeek-V2/V3): low-rank Q and KV projections via <code style={S.code}>q_lora_rank</code> and <code style={S.code}>kv_lora_rank</code>.</dd>
              <dt style={S.dt}>Sliding</dt>
              <dd style={S.dd}>Sliding-window attention (Mistral/Gemma/Llama 4 family); optionally hybridised with full attention via <code style={S.code}>full_attn_every_n</code>.</dd>
              <dt style={S.dt}>L-Hybrid</dt>
              <dd style={S.dd}>Linear-attention layers (gated linear / state-space style) interleaved with full-attention layers (Qwen3.5 family). Linear layers approximated as Q/K/V projections, depthwise convolution on Q/K, plus state update <code style={S.code}>K<sup>⊤</sup>V</code> and query application <code style={S.code}>Q·S</code>.</dd>
            </dl>
          </div>

          <div style={S.methodCol}>
            <h3 style={S.methodTitle}>GPU peak — per-GPU dense, TFLOPS</h3>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>GPU</th>
                  <th style={S.thNum}>BF16</th>
                  <th style={S.thNum}>FP8</th>
                  <th style={S.thNum}>FP4</th>
                  <th style={S.thNum}>HBM</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(GPUS).map(([k, g]) => (
                  <tr key={k}>
                    <td style={S.td}>{k}</td>
                    <td style={S.tdNum}>{g.bf16.toLocaleString()}</td>
                    <td style={S.tdNum}>{g.fp8.toLocaleString()}</td>
                    <td style={S.tdNum}>{g.fp4 ? g.fp4.toLocaleString() : '—'}</td>
                    <td style={S.tdNum}>{g.hbm} GB</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={S.methodNote}>
              Dense (non-sparse) tensor-core peak as reported by NVIDIA datasheets. Sparsity-2:4 doubles these figures but is rarely realised in training.
            </p>
          </div>

          <div style={S.methodCol}>
            <h3 style={S.methodTitle}>Reference MFU points</h3>
            <table style={S.table}>
              <thead>
                <tr><th style={S.th}>Setup</th><th style={S.thNum}>MFU</th></tr>
              </thead>
              <tbody>
                <tr><td style={S.td}>DSV3 / GB200 / FP8</td><td style={S.tdNum}>~23%</td></tr>
                <tr><td style={S.td}>DSV3 / GB300 / FP8</td><td style={S.tdNum}>~18%</td></tr>
              </tbody>
            </table>
            <p style={S.methodNote}>
              Reported in NVIDIA's Megatron-Core production training note. GB300 reaches 1,233 TFLOPS/GPU and GB200 reaches 1,048 TFLOPS/GPU on DeepSeek-V3-685B; GB300's lower MFU reflects a 56% peak-compute increase outpacing HBM and NVLink bandwidth — wall-clock is still ~17.6% faster.
            </p>
          </div>
        </div>

        <div style={S.methodFull}>
          <h3 style={S.methodTitle}>Real-world overhead</h3>
          <p style={S.bodyText}>
            This calculator yields a first-order, idealised estimate. Production training typically incurs a further <strong>10–30%</strong> overhead from communication bottlenecks (tensor / pipeline / expert / data parallelism), straggler and failure recovery, checkpoint I/O, MoE load imbalance, and FP8 kernel maturity. Reinforcement-learning pipelines incur additional cost from rollout / training alternation. <em>The reported time should be read as an optimistic floor, not a target.</em>
          </p>
        </div>

      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer style={S.footer}>
      <div>Generic LLM training time calculator. FLOPs verified against DeepSeek-V3 paper to 0.05pp.</div>
      <div style={S.footerSig}>⌬ for Rock · MiroMind</div>
    </footer>
  );
}

// ============================================================
// Styles — light theme
// ============================================================
const C = {
  bg: '#fafaf7',
  panel: '#ffffff',
  panel2: '#f5f5f0',
  border: '#e3e3dc',
  borderStrong: '#cdcdc4',
  text: '#1a1a1a',
  textMid: '#4a4a4a',
  textDim: '#777771',
  textMute: '#a3a39c',
  ink: '#0e0e0e',
  accent: '#2c4a6e',       // slate blue (academic)
  accentSoft: '#eef2f7',   // soft tint for active backgrounds
  accent2: '#1a4d3a',      // deep forest green
  accent3: '#722f37',      // muted burgundy
  highlight: '#f4ecd8',    // parchment
  good: '#1a8055',
  warn: '#a8651a',
};

const display = "'Fraunces', 'Newsreader', Georgia, serif";
const sans = "'Söhne', 'Inter Tight', 'Inter', system-ui, -apple-system, sans-serif";
const mono = "'JetBrains Mono', 'IBM Plex Mono', ui-monospace, Menlo, monospace";

const S = {
  root: {
    minHeight: '100vh',
    background: C.bg,
    color: C.text,
    fontFamily: sans,
    fontSize: 14,
    lineHeight: 1.5,
    position: 'relative',
  },
  bgGrid: {
    position: 'fixed', inset: 0,
    backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(0,0,0,0.04) 1px, transparent 0)',
    backgroundSize: '24px 24px',
    pointerEvents: 'none',
    zIndex: 0,
  },
  container: {
    maxWidth: 1320,
    margin: '0 auto',
    padding: '40px 28px 60px',
    position: 'relative',
    zIndex: 1,
  },

  // Header
  header: { marginBottom: 28 },
  tagRow: {
    display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16,
  },

  // Mode tabs
  modeTabs: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0,
    marginBottom: 24,
    border: `1px solid ${C.border}`,
    background: C.panel,
  },
  modeTab: {
    display: 'flex', alignItems: 'center', gap: 16,
    padding: '16px 22px',
    background: 'transparent',
    border: 'none',
    borderRight: `1px solid ${C.border}`,
    cursor: 'pointer',
    transition: 'all 0.15s',
    textAlign: 'left',
    color: C.textMid,
    position: 'relative',
  },
  modeTabActive: {
    background: C.accentSoft, color: C.accent,
    fontWeight: 600,
    borderBottom: `2px solid ${C.accent}`,
  },
  modeTabIcon: {
    fontFamily: display, fontSize: 28, fontWeight: 400, fontStyle: 'italic',
    flexShrink: 0,
    width: 36, height: 36,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    borderRadius: '50%',
    border: '1px solid currentColor',
    opacity: 0.6,
  },
  modeTabBody: {
    display: 'flex', flexDirection: 'column', gap: 2,
  },
  modeTabTitle: {
    fontFamily: display, fontSize: 19, fontStyle: 'italic',
    fontWeight: 500,
    letterSpacing: '-0.01em',
  },
  modeTabHint: {
    fontFamily: mono, fontSize: 10,
    letterSpacing: '0.1em', textTransform: 'uppercase',
    opacity: 0.7,
  },

  // Comparison box (in measure mode)
  cmpBox: {
    margin: '16px 0 18px',
    padding: 14,
    background: C.panel2,
    border: `1px solid ${C.border}`,
  },
  cmpTitle: {
    fontFamily: display, fontStyle: 'italic',
    fontSize: 15, fontWeight: 500, color: C.ink,
    marginBottom: 10,
  },
  cmpList: { display: 'flex', flexDirection: 'column', gap: 4 },
  cmpRow: {
    display: 'grid',
    gridTemplateColumns: '1fr auto auto',
    gap: 12, alignItems: 'baseline',
    fontFamily: mono, fontSize: 12,
    color: C.textMid,
    padding: '4px 8px',
    fontVariantNumeric: 'tabular-nums',
  },
  cmpRowMatch: {
    background: C.highlight,
    color: C.ink,
  },
  cmpLabel: {},
  cmpVal: {
    fontWeight: 600, color: C.ink,
    minWidth: 50, textAlign: 'right',
  },
  cmpDiff: {
    fontWeight: 600,
    minWidth: 60, textAlign: 'right',
  },
  tagAccent: {
    fontFamily: display, fontSize: 12, fontStyle: 'italic',
    color: C.accent, letterSpacing: '0.04em',
    fontVariant: 'small-caps', fontWeight: 500,
  },
  tagDivider: {
    fontFamily: display, color: C.textMute, fontSize: 14,
  },
  tagDim: {
    fontFamily: display, fontSize: 13, color: C.textDim,
    fontStyle: 'italic', letterSpacing: '0.01em',
  },
  title: {
    fontFamily: display,
    fontSize: 'clamp(40px, 6vw, 68px)',
    fontWeight: 400,
    letterSpacing: '-0.025em',
    lineHeight: 1.0,
    margin: '4px 0 0',
    fontStyle: 'italic',
    color: C.ink,
  },
  titleAccent: {
    fontFamily: display,
    fontStyle: 'normal',
    fontWeight: 400,
    color: C.accent,
    fontSize: '0.72em',
    marginLeft: 18,
    letterSpacing: '0.005em',
    fontVariant: 'small-caps',
  },
  subtitle: {
    fontFamily: display, fontSize: 15, color: C.text,
    maxWidth: 760, marginTop: 16, lineHeight: 1.65,
    letterSpacing: '0.005em',
  },

  // Preset bar
  presetBar: {
    display: 'flex', alignItems: 'center', gap: 16,
    padding: '12px 16px', marginBottom: 24,
    background: C.panel,
    border: `1px solid ${C.border}`,
    borderRadius: 0,
  },
  presetLabel: {
    display: 'flex', alignItems: 'center', gap: 8,
    fontFamily: mono, fontSize: 10, color: C.textDim,
    letterSpacing: '0.12em', textTransform: 'uppercase',
    paddingRight: 14, borderRight: `1px solid ${C.border}`,
    flexShrink: 0,
  },
  presetMark: {
    width: 6, height: 6, background: C.accent, borderRadius: '50%',
  },
  presetButtons: {
    display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center', flex: 1,
  },
  presetBtn: {
    fontFamily: sans, fontSize: 12, fontWeight: 500,
    padding: '6px 12px',
    background: 'transparent',
    color: C.textMid,
    border: `1px solid ${C.border}`,
    cursor: 'pointer', transition: 'all 0.12s',
  },
  presetBtnActive: {
    background: C.accentSoft, color: C.accent, borderColor: C.accent,
    fontWeight: 600,
  },
  customBadge: {
    fontFamily: mono, fontSize: 11, color: C.accent,
    marginLeft: 'auto', fontWeight: 600,
  },

  // Main grid
  mainGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.1fr)',
    gap: 20, marginBottom: 28,
  },
  rightCol: {
    display: 'flex', flexDirection: 'column', gap: 20,
  },

  // Panel
  panel: {
    background: C.panel,
    border: `1px solid ${C.border}`,
  },
  panelHead: {
    display: 'flex', alignItems: 'baseline', gap: 12,
    padding: '18px 24px',
    borderBottom: `1px solid ${C.border}`,
  },
  panelNum: {
    fontFamily: mono, fontSize: 10, color: C.textMute,
    letterSpacing: '0.12em',
  },
  panelTitle: {
    fontFamily: sans, fontSize: 12, fontWeight: 600,
    color: C.text, letterSpacing: '0.05em',
    textTransform: 'uppercase', margin: 0,
  },
  panelBody: { padding: '20px 24px' },

  // Field group
  fieldGroup: {
    marginBottom: 24, paddingBottom: 24,
    borderBottom: `1px dashed ${C.border}`,
  },
  fieldGroupTitle: {
    fontFamily: display, fontSize: 17, fontStyle: 'italic',
    fontWeight: 500, color: C.ink,
    marginBottom: 14,
  },

  // Field
  field: { marginBottom: 14 },
  fieldRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
    gap: 10, marginBottom: 14,
  },
  fieldLabel: {
    display: 'block',
    fontFamily: mono, fontSize: 10, color: C.textDim,
    letterSpacing: '0.06em', textTransform: 'uppercase',
    marginBottom: 5,
  },
  fieldBody: {},
  input: {
    width: '100%',
    padding: '8px 10px',
    background: C.panel,
    border: `1px solid ${C.borderStrong}`,
    color: C.ink,
    fontFamily: mono, fontSize: 13,
    outline: 'none',
    transition: 'border-color 0.12s',
    boxSizing: 'border-box',
  },
  slider: {
    width: '100%', marginTop: 8, accentColor: C.accent,
    cursor: 'pointer',
  },

  // Sub-section
  subSection: {
    marginTop: 12, padding: 14,
    background: C.panel2,
    border: `1px solid ${C.border}`,
  },
  subTitle: {
    fontFamily: mono, fontSize: 10, color: C.textDim,
    letterSpacing: '0.1em', textTransform: 'uppercase',
    marginBottom: 10,
  },
  note: {
    fontFamily: sans, fontSize: 12, color: C.textMid,
    fontStyle: 'italic', marginTop: 6,
  },

  // Segments
  segRow: {
    display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6,
  },
  seg: {
    flex: '1 1 auto', minWidth: 80,
    padding: '8px 10px',
    background: C.panel,
    border: `1px solid ${C.borderStrong}`,
    color: C.textMid,
    cursor: 'pointer', transition: 'all 0.12s',
    textAlign: 'left',
  },
  segActive: {
    background: C.accentSoft, color: C.accent, borderColor: C.accent,
    fontWeight: 600,
  },
  segLabel: {
    fontFamily: mono, fontSize: 12, fontWeight: 600,
    letterSpacing: '0.05em',
  },
  segHint: {
    fontFamily: sans, fontSize: 10, opacity: 0.7,
    marginTop: 2,
  },

  gpuSeg: {
    flex: '1 1 auto', minWidth: 64,
    padding: '8px 10px',
    background: C.panel,
    border: `1px solid ${C.borderStrong}`,
    color: C.textMid,
    cursor: 'pointer', transition: 'all 0.12s',
    textAlign: 'center',
  },
  gpuSegActive: {
    background: C.accentSoft, color: C.accent, borderColor: C.accent,
    fontWeight: 600,
  },
  precSeg: {
    flex: 1,
    padding: '8px 10px',
    background: C.panel,
    border: `1px solid ${C.borderStrong}`,
    color: C.textMid,
    cursor: 'pointer', transition: 'all 0.12s',
    textAlign: 'center',
  },
  precSegActive: {
    background: C.accentSoft, color: C.accent, borderColor: C.accent,
    fontWeight: 600,
  },
  precSegDisabled: {
    opacity: 0.35, cursor: 'not-allowed',
  },

  // Chips
  chipRow: { display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 },
  chip: {
    fontFamily: mono, fontSize: 11, padding: '3px 8px',
    background: 'transparent',
    border: `1px solid ${C.border}`,
    color: C.textDim, cursor: 'pointer',
  },
  chipActive: {
    background: C.accentSoft, color: C.accent, borderColor: C.accent,
    fontWeight: 600,
  },

  // Toggle (iOS-style switch)
  toggleRow: {
    display: 'flex', alignItems: 'center', gap: 12,
    cursor: 'pointer', userSelect: 'none',
    padding: '8px 0',
  },
  toggleInline: {
    display: 'inline-flex', alignItems: 'center', gap: 8,
    cursor: 'pointer', userSelect: 'none',
  },
  toggleBtn: {
    width: 36, height: 20, padding: 0,
    background: C.borderStrong,
    border: 'none', borderRadius: 999,
    cursor: 'pointer',
    transition: 'background 0.15s',
    position: 'relative',
    flexShrink: 0,
  },
  toggleBtnOn: { background: C.accent },
  toggleKnob: {
    display: 'block',
    width: 16, height: 16,
    background: '#fff',
    borderRadius: '50%',
    position: 'absolute',
    top: 2, left: 2,
    transition: 'transform 0.15s',
    boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
  },
  toggleKnobOn: { transform: 'translateX(16px)' },
  toggleText: {},
  toggleLabel: {
    fontFamily: sans, fontSize: 13, fontWeight: 500, color: C.text,
  },
  toggleHint: {
    fontFamily: sans, fontSize: 11, color: C.textDim, marginTop: 2,
  },

  // MFU reference
  mfuRefBox: {
    marginTop: 10, padding: 10,
    background: C.panel2,
    border: `1px solid ${C.border}`,
  },
  mfuRefTitle: {
    fontFamily: mono, fontSize: 9, color: C.textDim,
    letterSpacing: '0.1em', textTransform: 'uppercase',
    marginBottom: 6,
  },
  mfuRefList: { display: 'flex', flexDirection: 'column', gap: 3 },
  mfuRefItem: {
    display: 'flex', justifyContent: 'space-between',
    fontFamily: mono, fontSize: 11,
    color: C.textMid,
    padding: '2px 4px',
  },
  mfuRefItemMatch: {
    background: C.highlight,
    color: C.ink,
    fontWeight: 600,
  },
  mfuRefLabel: {},
  mfuRefVal: { fontWeight: 600 },

  // BigStat
  bigStat: {
    padding: '20px 22px',
    background: C.highlight,
    borderLeft: `3px solid ${C.accent}`,
    marginBottom: 18,
  },
  bigStatLabel: {
    fontFamily: mono, fontSize: 10,
    letterSpacing: '0.15em',
    color: C.warn,
  },
  bigStatValue: {
    fontFamily: display, fontStyle: 'italic',
    fontSize: 'clamp(32px, 4.5vw, 48px)', fontWeight: 400,
    color: C.ink, lineHeight: 1, marginTop: 4,
    letterSpacing: '-0.02em',
  },
  bigStatSub: {
    fontFamily: mono, fontSize: 12, color: C.textMid,
    marginTop: 8,
  },

  // Stat grid
  statGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
    gap: 1,
    background: C.border,
    border: `1px solid ${C.border}`,
    marginBottom: 18,
  },
  stat: {
    background: C.panel, padding: '12px 14px',
  },
  statLabel: {
    fontFamily: mono, fontSize: 9,
    letterSpacing: '0.08em',
    color: C.textMute, textTransform: 'uppercase',
  },
  statValue: {
    fontFamily: mono, fontSize: 16, fontWeight: 500,
    color: C.ink, marginTop: 3,
  },

  // Breakdown
  breakdown: {
    marginTop: 18, paddingTop: 16,
    borderTop: `1px dashed ${C.border}`,
  },
  brRow: {
    display: 'grid',
    gridTemplateColumns: '1.4fr 2fr 1fr',
    alignItems: 'center', gap: 12,
    padding: '6px 0',
  },
  brRowSub: { opacity: 0.55, paddingLeft: 12 },
  brLabel: {
    fontFamily: sans, fontSize: 12, color: C.text,
  },
  brBarOuter: {
    height: 4, background: C.panel2,
    border: `1px solid ${C.border}`,
    position: 'relative',
  },
  brBarFill: {
    height: '100%', background: C.accent2,
    transition: 'width 0.3s',
  },
  brBarFillSub: { background: C.textMute },
  brVal: {
    display: 'flex', justifyContent: 'flex-end', gap: 8,
    fontFamily: mono, fontSize: 12, color: C.text,
    fontVariantNumeric: 'tabular-nums',
  },
  brPct: { color: C.textDim, fontSize: 11 },

  // Formula section (per-component breakdown)
  formulaSection: {
    marginTop: 22, paddingTop: 18,
    borderTop: `1px dashed ${C.border}`,
  },
  formulaToggle: {
    display: 'flex', alignItems: 'center', gap: 8,
    width: '100%', padding: 0, margin: 0,
    background: 'transparent', border: 'none', cursor: 'pointer',
    textAlign: 'left',
  },
  formulaCaret: {
    fontSize: 10, color: C.textDim,
    transition: 'transform 0.15s ease',
    display: 'inline-block',
  },
  formulaToggleHint: {
    marginLeft: 'auto', fontSize: 11, color: C.textDim,
  },
  formulaGroup: {
    marginTop: 14, marginBottom: 10,
    paddingLeft: 12,
    borderLeft: `2px solid ${C.accent2}`,
  },
  formulaGroupTitle: {
    fontFamily: display, fontSize: 14, fontStyle: 'italic',
    fontWeight: 500, color: C.ink,
    marginBottom: 6,
  },
  formulaRow: {
    display: 'grid',
    gridTemplateColumns: '120px 1fr',
    gap: 12, alignItems: 'baseline',
    padding: '3px 0',
    borderBottom: `1px dotted ${C.border}`,
  },
  formulaName: {
    fontFamily: sans, fontSize: 11,
    color: C.textMid, fontWeight: 500,
  },
  formulaBody: {
    display: 'flex', flexWrap: 'wrap',
    alignItems: 'baseline',
    gap: 6,
    fontFamily: mono, fontSize: 11,
    fontVariantNumeric: 'tabular-nums',
  },
  formulaSym: {
    color: C.accent2,
  },
  formulaEq: {
    color: C.textMute,
    fontWeight: 600,
  },
  formulaSub: {
    color: C.textMid,
  },
  formulaVal: {
    color: C.ink, fontWeight: 600,
  },
  formulaNote: {
    fontStyle: 'italic', color: C.textDim,
  },
  formulaLegend: {
    fontFamily: mono, fontSize: 10,
    color: C.textDim, marginTop: 14,
    fontStyle: 'italic',
    lineHeight: 1.6,
  },

  // Methodology
  method: {
    background: C.panel,
    border: `1px solid ${C.border}`,
    marginBottom: 28,
  },
  methodBody: {
    padding: '20px 24px 28px',
  },
  methodGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    columnGap: 36,
    rowGap: 26,
  },
  methodCol: { minWidth: 0 },
  methodFull: {
    marginTop: 26, paddingTop: 22,
    borderTop: `1px dashed ${C.border}`,
  },
  methodTitle: {
    fontFamily: display, fontSize: 17, fontStyle: 'italic',
    fontWeight: 500, color: C.ink,
    marginTop: 0, marginBottom: 12,
    paddingBottom: 6,
    borderBottom: `1px solid ${C.border}`,
    letterSpacing: '0.01em',
  },
  methodNote: {
    fontFamily: display, fontSize: 13, color: C.textDim,
    fontStyle: 'italic', marginTop: 10, lineHeight: 1.55,
  },
  bullets: {
    margin: 0, paddingLeft: 20,
    fontFamily: display, fontSize: 14, color: C.text,
    lineHeight: 1.7, letterSpacing: '0.005em',
  },
  bodyText: {
    fontFamily: display, fontSize: 14, color: C.text,
    lineHeight: 1.7, letterSpacing: '0.005em',
    margin: 0, textIndent: '1.5em',
  },
  dl: {
    margin: 0,
    fontFamily: display, fontSize: 14, color: C.text,
    lineHeight: 1.55,
    display: 'grid',
    gridTemplateColumns: '70px 1fr',
    columnGap: 14, rowGap: 8,
  },
  dt: {
    fontFamily: sans, fontWeight: 600, fontSize: 11,
    color: C.accent, letterSpacing: '0.08em',
    textTransform: 'uppercase',
    paddingTop: 3,
  },
  dd: {
    margin: 0, fontFamily: display, fontSize: 14,
    color: C.text, lineHeight: 1.55,
  },
  refs: {
    margin: 0, paddingLeft: 22,
    fontFamily: display, fontSize: 13, color: C.text,
    lineHeight: 1.6, letterSpacing: '0.005em',
  },
  refAuthor: {
    fontFamily: display, fontVariant: 'small-caps',
    letterSpacing: '0.04em',
    color: C.text, fontWeight: 500,
  },
  refTitle: {
    fontFamily: display, fontStyle: 'italic',
    color: C.ink,
  },
  refMeta: {
    fontFamily: sans, fontSize: 12, color: C.textDim,
  },
  code: {
    fontFamily: mono, fontSize: 11,
    background: C.panel2,
    padding: '1px 5px',
    color: C.accent2,
    border: `1px solid ${C.border}`,
    borderRadius: 2,
  },
  table: {
    width: '100%', borderCollapse: 'collapse',
    fontFamily: mono, fontSize: 12,
  },
  th: {
    textAlign: 'left',
    padding: '6px 8px',
    borderBottom: `1px solid ${C.borderStrong}`,
    color: C.textDim, fontWeight: 500,
    fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase',
  },
  thNum: {
    textAlign: 'right',
    padding: '6px 8px',
    borderBottom: `1px solid ${C.borderStrong}`,
    color: C.textDim, fontWeight: 500,
    fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase',
  },
  td: {
    padding: '6px 8px',
    borderBottom: `1px solid ${C.border}`,
    color: C.text,
    fontVariantNumeric: 'tabular-nums',
  },
  tdNum: {
    padding: '6px 8px',
    borderBottom: `1px solid ${C.border}`,
    color: C.text,
    fontVariantNumeric: 'tabular-nums',
    textAlign: 'right',
  },

  // Footer
  footer: {
    marginTop: 28, paddingTop: 18,
    borderTop: `1px solid ${C.border}`,
    fontFamily: mono, fontSize: 11, color: C.textDim,
    display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12,
  },
  footerSig: { letterSpacing: '0.04em' },
};

// Inject Google Font
if (typeof document !== 'undefined' && !document.getElementById('calc-v2-fonts')) {
  const el = document.createElement('style');
  el.id = 'calc-v2-fonts';
  el.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..700&family=Inter+Tight:wght@300..600&family=JetBrains+Mono:wght@400..600&display=swap');
    button:focus-visible, input:focus-visible {
      outline: 2px solid ${C.accent};
      outline-offset: 1px;
    }
    input[type=number]::-webkit-inner-spin-button { opacity: 0.4; }
    @media (max-width: 980px) {
      [data-grid] { grid-template-columns: 1fr !important; }
    }
  `;
  document.head.appendChild(el);
}

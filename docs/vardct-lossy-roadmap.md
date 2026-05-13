# VarDCT Lossy Roadmap

This document tracks the still-image lossy JPEG XL work requested in
`goals/goal-01.md`. The scope is VarDCT/lossy still decode without animation,
JPEG reconstruction, libjxl FFI, or a public API redesign. The checked-in
`reference/libjxl` tools may be used only as test oracles.

## Current Gap Map

Audit date: 2026-05-13.

### Implemented or Partly Implemented

- Container/codestream/frame parsing records VarDCT plans in
  `crates/jxl-codec/src/codestream.rs` and exposes them through `jxl::inspect`.
- VarDCT section metadata, global metadata, AC global metadata, DC group
  metadata, AC group metadata, and pass/group payload metadata are parsed in
  `crates/jxl-codec/src/vardct.rs`.
- Quantizer metadata, DC dequant metadata, block context maps, coefficient
  order, AC entropy metadata probes, AC token decoding, per-channel dequant
  matrices, and DC coefficient diagnostics have Rust implementations.
- Spatial reconstruction supports the common DCT families covered by
  `spatial_transform_for_strategy`, default dequant matrices for strategies
  0 through 26, identity/DCT2/DCT4/DCT8 rectangular transforms, AFV helpers,
  and larger rectangular DCT strategies.
- XYB assembly, DC-only assembly, per-pass assembly, RGB/YCbCr/direct-RGB
  conversion helpers, 8-bit and 16-bit sRGB presentation helpers, gaborish,
  EPF entry points, frame upsampling, and limited chroma upsampling are present.
- Public `jxl` APIs route VarDCT through `decode`, `decode_rgba8`,
  `decode_rgba16`, `decode_channels`, ROI decode, and progressive pass decode
  where currently supported.
- VarDCT alpha and non-alpha extra channels stored in modular side streams are
  exposed by `decode_channels` when all channel chunks are decoded and filled.
- Existing generated tests use `reference/libjxl/build-rs-oracle/tools/cjxl`
  and `djxl` when available and skip gracefully when the tools are missing.
- Existing public/API tests cover generated RGB, grayscale, YCbCr, alpha,
  alpha16, depth, combined alpha/depth, ROI, progressive pass, frame
  upsampling, and modular-side-stream extra-channel workflows.
- Existing codec-level tests cover global and AC cursors, pass payloads,
  AC prefix paths, DC diagnostics, transform variants, quant matrices,
  gaborish/EPF toggles, chroma subsampling, and multiple VarDCT fixture shapes.

### Known Gaps

- The roadmap file itself was missing before this milestone.
- Current public oracle comparisons are primarily regression checks for the
  current implementation. Several generated lossy RGB/gray tests still allow
  large errors against `djxl`, so the milestone 3 acceptance tolerance
  (RGBA8 max absolute channel error <= 2 and mean <= 0.25) is not yet met for
  common generated images.
- Tiny fixture coverage is not yet explicit as a matrix of 1x1, 2x2, 8x8,
  16x16, 32x32, grayscale, RGB, alpha, and simple gradients.
- Full JPEG XL color management is not implemented. Public output explicitly
  supports only the currently handled sRGB/XYB/direct/YCbCr paths and should
  keep returning precise `Error::Unsupported` messages for unsupported
  wide-gamut/custom ICC/full color-management cases.
- Some production diagnostic helpers still contain invariant `expect` calls in
  code paths that are intended to run only after prior validation. These should
  be revisited during milestone 2 and converted to explicit errors if they can
  be reached from malformed input.
- 16-bit VarDCT presentation output exists, but high-bit-depth lossy
  conformance still needs oracle-backed tolerances and targeted grayscale tests.
- Up to 8 total decoded channels are not yet documented by a focused generated
  VarDCT fixture. Existing tests cover alpha, depth, and combined alpha/depth
  side streams, but not an explicit 8-channel workflow.
- Unsupported transform/filter/extra-channel layouts are not comprehensively
  covered by tests that assert exact `Error::Unsupported` messages.

## Fixture and Oracle Harness

Existing harnesses:

- `crates/jxl-codec/tests/fixtures.rs` includes codec-internal VarDCT tests
  with generated sources, reference `cjxl`/`djxl` helpers, PPM/PAM parsing, and
  sRGB metric summaries.
- `crates/jxl/src/lib.rs` includes public API tests with generated sources,
  reference `cjxl`/`djxl` helpers, `decode`, `decode_rgba8`, `decode_rgba16`,
  `decode_channels`, ROI, and progressive pass checks.
- Reference tool discovery uses `JXL_RS_REFERENCE_CJXL`,
  `JXL_RS_REFERENCE_DJXL`, and the default checked-in paths under
  `reference/libjxl/build-rs-oracle/tools`.

Harness gaps to close:

- Add a named tiny-fixture matrix for generated lossy VarDCT cases. The first
  matrix should include 1x1, 2x2, 8x8, 16x16, and 32x32 RGB gradients, plus
  grayscale and alpha variants where `cjxl` emits VarDCT.
- Document tolerances in each oracle test. Keep loose regression tolerances only
  where they describe known incomplete decode quality, and add failing-work
  notes instead of presenting them as conformance.
- Add acceptance-level tests as implementation quality improves. The target for
  milestone 3 is RGBA8 max absolute channel error <= 2 and mean absolute error
  <= 0.25 against `djxl`.

## Milestone Status

### Milestone 1: Gap Map and Fixture/Oracle Harness

Status: complete.

Progress:

- Audited `crates/jxl-codec/src/vardct.rs`, `modular.rs`, `frame.rs`,
  `frame_data.rs`, `transform.rs`, public output paths in `crates/jxl/src/lib.rs`,
  and existing generated fixture tests.
- Added this roadmap and gap map.
- Added `decode_tiny_generated_var_dct_matrix_against_oracle_when_available`
  in `crates/jxl/src/lib.rs`. It generates 1x1, 2x2, 8x8, 16x16, and 32x32
  RGB gradients, plus 16x16 grayscale and RGBA cases, encodes them with `cjxl`,
  decodes oracle output with `djxl`, skips when reference tools are missing,
  and compares public Rust output against the oracle with documented coarse
  milestone-1 regression thresholds.

Remaining:

- Later milestones still need acceptance-level lossy tolerances; the new tiny
  matrix deliberately records current behavior and is not a conformance claim.

### Milestone 2: Precise VarDCT Failure Modes

Status: in progress.

Focus:

- Replace any reachable vague VarDCT reconstruction failures or invariant
  panics with `Error::Unsupported` or `Error::InvalidCodestream`.
- Add tests for unsupported strategies, transforms, color spaces, filters, and
  extra-channel layouts.

Progress:

- Converted reachable VarDCT DC coefficient diagnostics from invariant
  `expect` calls to explicit `Error::Unsupported("VarDCT DC coefficients")`
  or propagated invalid-codestream errors.
- Converted 8x8 DCT reconstruction from an invariant `expect` to a `Result`
  path.
- Replaced dequant interpolation `unwrap` calls with explicit invalid
  codestream errors, and added a unit regression for empty/invalid band inputs.
- Added unit coverage pinning precise errors for unsupported chroma upsampling,
  unsupported frame upsampling, invalid custom upsampling weight counts, and
  invalid AC strategies.

### Milestone 3: Baseline VarDCT RGB8 Still Decode

Status: pending.

Focus:

- Bring common 8-bit RGB/RGBA/grayscale lossy still images produced by `cjxl`
  within the documented `djxl` tolerance.
- Cover naked codestream and container inputs, default orientation, supported
  non-default orientation, no alpha, and simple alpha.

### Milestone 4: DC Path and Low-Frequency Correctness

Status: pending.

Focus:

- Verify DC group decoding, DC dequantization, color correlation, and LF image
  assembly against reference behavior and keep progressive pass behavior green.

### Milestone 5: AC Coefficient Decoding and Dequantization

Status: pending.

Focus:

- Complete common AC token decode, coefficient ordering, context selection,
  nonzero counts, sign/unpack logic, quant matrices, dequantization, and
  malformed coefficient-stream tests.

### Milestone 6: Inverse Transforms and Block Strategies

Status: pending.

Focus:

- Fill missing transform strategies emitted by common `cjxl` settings and keep
  unsupported strategies explicit.

### Milestone 7: XYB Inverse and Color Output

Status: pending.

Focus:

- Improve standard XYB-to-RGB fidelity, respect parsed opsin parameters where
  practical, and keep unsupported color management explicit.

### Milestone 8: 16-Bit Output Paths

Status: pending.

Focus:

- Add oracle-backed 16-bit grayscale and RGB/RGBA presentation tests and ensure
  `decode_channels` does not downconvert supported high-bit-depth channels.

### Milestone 9: Extra Channels and Up to 8-Channel Workflows

Status: pending.

Focus:

- Add explicit generated fixtures for RGB plus alpha, grayscale plus alpha,
  RGB plus multiple non-alpha extra channels where fixture generation supports
  it, and up to 8 total decoded channels through `decode_channels`.

### Milestone 10: Filters, Upsampling, and Common Lossy Conformance

Status: pending.

Focus:

- Finish common gaborish, EPF, chroma subsampling/YCbCr, frame upsampling,
  custom upsampling kernels, small-image, and partial-group cases with
  documented tolerances.

## Progress Notes

- 2026-05-13: Completed the initial audit and created this roadmap. Current
  tests already exercise many generated VarDCT workflows, but acceptance-level
  lossy fidelity and explicit tiny-fixture coverage remain open.
- 2026-05-13: Added an explicit tiny generated VarDCT oracle matrix covering
  RGB gradients from 1x1 through 32x32, grayscale, and RGBA. The focused test
  and all required milestone gates pass locally with the checked-in reference
  tools:
  `cargo fmt --all -- --check`, `cargo check --workspace`,
  `cargo test -p jxl-codec --lib`, and `cargo test --workspace`.
- 2026-05-13: Started milestone 2 by removing reachable production VarDCT
  `expect`/`unwrap` cases in DC diagnostics, DCT8 reconstruction, and dequant
  interpolation. The focused `dequant_interpolation_rejects_empty_bands` unit
  test passes, production `crates/jxl-codec/src/vardct.rs` no longer has
  non-test `unwrap`/`expect` hits, and the required milestone gates pass:
  `cargo fmt --all -- --check`, `cargo check --workspace`,
  `cargo test -p jxl-codec --lib`, and `cargo test --workspace`.
- 2026-05-13: Added `unsupported_vardct_paths_report_precise_errors` to lock
  down current precise `Unsupported`/`InvalidCodestream` errors for VarDCT
  upsampling and strategy edge cases. The required gates pass after this added
  coverage: `cargo fmt --all -- --check`, `cargo check --workspace`,
  `cargo test -p jxl-codec --lib`, and `cargo test --workspace`.

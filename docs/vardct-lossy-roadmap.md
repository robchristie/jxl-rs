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

- Current public oracle comparisons are primarily regression checks for the
  current implementation. Several generated lossy RGB/gray tests still allow
  large errors against `djxl`, so the milestone 3 acceptance tolerance
  (RGBA8 max absolute channel error <= 2 and mean <= 0.25) is not yet met for
  common generated images.
- The generated JPEG-origin YCbCr/None VarDCT fixtures exercise libjxl's
  JPEG-transcode-style color-transform paths and keep loose regression metrics.
  They are useful coverage for those color paths, but they are not evidence
  that JPEG reconstruction is supported or that milestone-3 RGB8 conformance is
  met.
- Full JPEG XL color management is not implemented. Public output explicitly
  supports only the currently handled sRGB/XYB/direct/YCbCr paths and should
  keep returning precise `Error::Unsupported` messages for unsupported
  wide-gamut/custom ICC/full color-management cases.
- Some production diagnostic helpers still contain invariant `expect` calls in
  code paths that are intended to run only after prior validation. These should
  be revisited during milestone 2 and converted to explicit errors if they can
  be reached from malformed input.
- 16-bit VarDCT presentation output exists, but high-bit-depth lossy
  conformance remains above the milestone-8 RGBA16 target tolerance for the
  generated grayscale16 fixture.
- Up to 8 total decoded channels are documented by a focused generated VarDCT
  fixture that keeps RGB presentation output separate from non-alpha
  `decode_channels` side-stream output.
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
- Replaced the remaining non-test `unreachable!()` sites in the audited public
  alpha upsampling, VarDCT upsampling/quant-matrix, and modular pass bracket
  paths with explicit `Result` errors.
- Added public API unit coverage for invalid alpha upsampling weights.

### Milestone 3: Baseline VarDCT RGB8 Still Decode

Status: in progress.

Focus:

- Bring common 8-bit RGB/RGBA/grayscale lossy still images produced by `cjxl`
  within the documented `djxl` tolerance.
- Cover naked codestream and container inputs, default orientation, supported
  non-default orientation, no alpha, and simple alpha.

### Milestone 4: DC Path and Low-Frequency Correctness

Status: in progress.

Focus:

- Verify DC group decoding, DC dequantization, color correlation, and LF image
  assembly against reference behavior and keep progressive pass behavior green.

Progress:

- Added the libjxl adaptive DC smoothing stage to final and DC-only 4:4:4
  VarDCT reconstruction when the frame does not set `SkipAdaptiveDCSmoothing`
  or `UseDcFrame`. Added focused unit coverage for the smoothing kernel. The
  generated multigroup public fixture improves to max absolute error `14` and
  sum absolute error `764_386`, and the non-default QF codec fixture sum drops
  to `62_571_873`, but these are still above milestone-3 acceptance tolerance.

### Milestone 5: AC Coefficient Decoding and Dequantization

Status: in progress.

Focus:

- Complete common AC token decode, coefficient ordering, context selection,
  nonzero counts, sign/unpack logic, quant matrices, dequantization, and
  malformed coefficient-stream tests.

Progress:

- Matched libjxl's default DCT4X8/DCT8X4 quant-table construction by deriving
  the 8x8 table from a 4x8 base and duplicating coefficient rows, instead of
  interpolating an 8x8 table directly.

### Milestone 6: Inverse Transforms and Block Strategies

Status: pending.

Focus:

- Fill missing transform strategies emitted by common `cjxl` settings and keep
  unsupported strategies explicit.

### Milestone 7: XYB Inverse and Color Output

Status: in progress.

Focus:

- Improve standard XYB-to-RGB fidelity, respect parsed opsin parameters where
  practical, and keep unsupported color management explicit.

### Milestone 8: 16-Bit Output Paths

Status: in progress.

Focus:

- Add oracle-backed 16-bit grayscale and RGB/RGBA presentation tests and ensure
  `decode_channels` does not downconvert supported high-bit-depth channels.

Progress:

- Added public generated grayscale16 VarDCT coverage for `decode_rgba16`
  against `djxl`. The current output is still a loose regression snapshot,
  not milestone-8 conformance: max absolute channel error is `45_211` and sum
  absolute error is `86_222_328`, far above the target tolerance.

### Milestone 9: Extra Channels and Up to 8-Channel Workflows

Status: in progress.

Focus:

- Add explicit generated fixtures for RGB plus alpha, grayscale plus alpha,
  RGB plus multiple non-alpha extra channels where fixture generation supports
  it, and up to 8 total decoded channels through `decode_channels`.

Progress:

- Added public generated VarDCT coverage for RGB plus five non-alpha modular
  side-stream extra channels. `decode_channels` now has explicit regression
  coverage for 8 total decoded channels, while `decode`/`decode_rgba8` continue
  to expose only RGB/RGBA presentation output for that fixture.

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
- 2026-05-13: Removed the remaining non-test `unreachable!()` hits found in the
  audited public/VarDCT/modular reconstruction files and added
  `alpha_upsampling_weights_report_precise_errors`. Focused public and codec
  tests pass, and the required gates pass: `cargo fmt --all -- --check`,
  `cargo check --workspace`, `cargo test -p jxl-codec --lib`, and
  `cargo test --workspace`.
- 2026-05-13: Advanced the DC/LF path by threading frame color transform into
  VarDCT global metadata parsing, matching libjxl's non-XYB default color
  correlation, and applying 4:4:4 DC color correlation during VarDCT DC
  coefficient reconstruction. Updated codec/public oracle snapshots for the
  corrected low-frequency output. The required gates pass after this change:
  `cargo fmt --all -- --check`, `cargo check --workspace`,
  `cargo test -p jxl-codec --lib`, and `cargo test --workspace`.
- 2026-05-13: Advanced the XYB inverse path by switching the default VarDCT
  XYB-to-linear-RGB conversion to the libjxl formula (`gamma_b = b - cbrt_bias`)
  instead of the older diagnostic negative-B variant. Codec and public oracle
  snapshots now use the lower-error B-minus-bias path; common generated RGB
  VarDCT still remains above milestone-3 tight tolerance and needs further AC,
  filtering, and color-output work.
- 2026-05-13: Advanced AC/DC spatial reconstruction by matching libjxl's scaled
  VarDCT inverse-DCT normalization instead of compensating only the DC
  coefficient with a multiplier of `8.0`. Large DCT strategies now derive their
  full low-frequency coefficient rectangle from the decoded DC image using the
  same reinterpreting-DCT resampling factors as libjxl. This substantially
  improves several generated XYB/RGBA oracle metrics (for example the common
  public RGB fixture sum-absolute error drops from `7_696_330` to about
  `2.7M`, and the multigroup fixture drops from `174_532_848` to `940_536`), but
  JPEG-color-transform and subsampled YCbCr fixtures still have large errors and
  milestone-3 conformance remains open. Focused codec/public fixture tests pass
  with updated regression snapshots.
- 2026-05-13: Aligned AC coefficient decoding more closely with libjxl by
  applying each frame pass shift to decoded AC coefficients and by using the
  shifted block position for subsampled chroma quant-field entropy contexts
  while preserving the full-resolution DC context. Added unit coverage for
  pass-shift scaling and checked overflow behavior. Focused RGB JPEG and YCbCr
  4:2:0/axis/odd-dimension VarDCT fixture tests pass; the common generated
  JPEG-color-transform and subsampled YCbCr oracle metrics remain above the
  milestone-3 tight tolerance, so conformance work remains open.
- 2026-05-13: Rechecked the JPEG-origin fixture mode against the local libjxl
  `cjxl` help. The current `--allow_jpeg_reconstruction=0` argument does not
  disable JPEG-transcode-style VarDCT color-transform paths in this build; the
  newer `--lossless_jpeg=0` path makes these particular inputs encode as
  modular/pixel fixtures instead. Kept the existing fixtures as explicit loose
  YCbCr/None color-transform coverage and documented that they are not JPEG
  reconstruction support or milestone-3 acceptance evidence.
- 2026-05-13: Added
  `decode_channels_exposes_eight_generated_var_dct_channels_when_available`.
  The test generates a VarDCT RGB image with Depth, SelectionMask, CFA,
  Thermal, and Optional side-stream extra channels. It verifies
  `decode_channels` exposes all 8 channels with exact extra-channel samples,
  while public RGB/RGBA presentation output ignores the non-alpha extras by
  design. Focused test passes with the checked-in reference `cjxl`.
- 2026-05-13: Added
  `decode_rgba16_supports_generated_gray16_var_dct_when_available` as an
  oracle-backed milestone-8 regression fixture. It confirms the public RGBA16
  path runs for generated grayscale16 VarDCT output, but the captured metrics
  remain far outside the intended RGBA16 conformance tolerance, so the
  grayscale high-bit-depth reconstruction path still needs algorithmic work.
- 2026-05-13: Implemented adaptive DC smoothing for final/DC-only 4:4:4
  VarDCT reconstruction, matching libjxl's placement after DC dequantization
  and before AC reconstruction for frames that do not skip the feature. Added a
  focused codec unit test for the smoothing kernel. The generated multigroup
  public fixture improved from max/sum `18`/`940_536` to `14`/`764_386`, and
  the non-default QF codec fixture sum moved from `62_572_812` to
  `62_571_873`, but the remaining RGB8 and gray16 fidelity gaps still point at
  AC/dequant/transform/color-output work.
- 2026-05-13: Started milestone 7 custom opsin coverage by threading parsed
  VarDCT quant biases from `OpsinInverseMatrix` through opsin parameters and
  AC dequantization. Default fixtures are expected to stay unchanged, but
  custom-transform files now use their encoded quant-bias values instead of
  the hard-coded defaults.
- 2026-05-13: Advanced milestone 5 by correcting default DCT4X8/DCT8X4
  dequant matrices to match libjxl's 4x8-base row-duplication layout. This
  changes affected regression snapshots only slightly: the common RGB public
  fixture sum is now `2_742_408`, and the non-default QF codec fixture sum is
  now `62_570_488`. Milestone-3 fidelity remains far above the tight
  acceptance tolerance.

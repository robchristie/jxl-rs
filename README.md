# jxl-rs

An idiomatic Rust JPEG XL implementation in progress.

The current codebase is deliberately focused on the foundation every conforming
decoder needs:

- JPEG XL naked codestream and container signature detection.
- JPEG XL container box parsing, including extended-size boxes.
- Codestream extraction from `jxlc` and ordered `jxlp` boxes.
- JPEG XL size-header parsing with the same field encoding as libjxl.
- Top-level image metadata parsing: orientation, intrinsic/preview/animation
  flags, bit depth, extra channels, XYB/original-profile flag, color encoding,
  and tone mapping.
- Custom transform-data parsing after basic info, including optional custom
  opsin inverse matrix and upsampling kernel weights.
- ICC profile parsing for codestreams that signal `want_icc`, including the
  JPEG XL ICC entropy stream and predictor reversal.
- First-frame header parsing: frame type, modular/VarDCT encoding selection,
  color transform, frame crop/origin, passes, blending, animation timing,
  loop-filter parameters, and group layout.
- First-frame data table-of-contents parsing and section payload traversal,
  including DC/global and AC group section classification.
- First-frame modular global metadata parsing for frames without optional
  patch/spline/noise preambles, including the DC dequant preamble, global MA
  tree, weighted-predictor header, and modular transforms.
- First-frame modular DC/AC group stream header parsing, including stream ID
  derivation and local MA tree metadata when a group does not use the global
  tree.
- First-frame modular channel planning after global metadata transforms, with
  per-group channel rectangles for future residual token decoding.
- A first modular residual token decoder for planned group channels, including
  weighted predictor properties/state and sample decoding for simple modular
  fixtures such as `pq_gradient.jxl`.
- A small public inspection and still-image decode API through the `jxl` crate,
  including modular raw channel output, modular RGB/RGBA output, and supported
  VarDCT RGB/RGBA output with post-reconstruction ROI cropping and progressive
  AC pass selection.
- A `jxlinfo-rs` CLI for metadata inspection.
- A `jxl-decode-rs` CLI for supported still-image decode to RGBA PAM.
- Fixture tests against `reference/libjxl/testdata`, with optional comparison
  to the built libjxl `jxlinfo` reference tool.

The decoder remains incomplete. Current VarDCT output is a reconstruction
convenience path and does not yet implement full JPEG XL color management,
orientation handling, low-memory ROI, animation, or JPEG reconstruction.

## Workspace

- `crates/jxl-codec`: core codestream and container primitives.
- `crates/jxl`: public Rust-native API.
- `crates/jxl-cli`: command-line tools.
- `reference/libjxl`: C++ reference implementation used only as an oracle in
  tests and development, never through FFI.

## Usage

```sh
cargo run -p jxl-cli --bin jxlinfo-rs -- reference/libjxl/testdata/jxl/splines.jxl
```

Decode a supported still image to RGBA PAM, 8-bit by default:

```sh
cargo run -p jxl-cli --bin jxl-decode-rs -- input.jxl output.pam
```

Decode 16-bit RGBA PAM:

```sh
cargo run -p jxl-cli --bin jxl-decode-rs -- input.jxl output.pam --rgba16
```

Decode a region of interest:

```sh
cargo run -p jxl-cli --bin jxl-decode-rs -- input.jxl output.pam --roi 10,20,320,240
```

Write PAM to stdout for pipelines:

```sh
cargo run -p jxl-cli --bin jxl-decode-rs -- input.jxl - --roi 10,20,320,240 > crop.pam
```

For VarDCT images with progressive AC passes, select exactly one AC pass for
preview-style RGB/RGBA output:

```sh
cargo run -p jxl-cli --bin jxl-decode-rs -- input.jxl preview.pam --vardct-pass 0
```

`--vardct-pass` is only valid for VarDCT RGB/RGBA output. It does not merge
earlier or later progressive passes. Output from `jxl-decode-rs` is always RGBA
PAM; use `--rgba8`/`--bits 8` for 8-bit samples and `--rgba16`/`--bits 16` for
16-bit samples.

## Reference Tools

The tests can compare parsed dimensions against the reference `jxlinfo` tool.
The default path is:

```text
reference/libjxl/build-rs-oracle/tools/jxlinfo
```

To rebuild the reference tools:

```sh
cmake -S reference/libjxl -B reference/libjxl/build-rs-oracle -G Ninja \
  -DJPEGXL_ENABLE_PLUGINS=OFF \
  -DJPEGXL_ENABLE_MANPAGES=OFF \
  -DJPEGXL_ENABLE_BENCHMARK=OFF \
  -DJPEGXL_ENABLE_EXAMPLES=OFF \
  -DJPEGXL_ENABLE_JNI=OFF \
  -DJPEGXL_ENABLE_DEVTOOLS=OFF \
  -DJPEGXL_ENABLE_VIEWERS=OFF
cmake --build reference/libjxl/build-rs-oracle --target jxlinfo djxl cjxl -j 8
```

Set `JXL_RS_REFERENCE_JXLINFO=/path/to/jxlinfo` to use a different oracle
binary.

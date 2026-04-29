# Reference Trace Tool

This directory contains a narrow libjxl-backed oracle for the generated split
VarDCT fixture used by `crates/jxl-codec/tests/fixtures.rs`.

Build it against the local reference build:

```sh
tools/reference-trace/build.sh
```

Then point the optional fixture hook at the resulting binary:

```sh
JXL_RS_REFERENCE_TRACE=target/reference-trace/jxl_vardct_trace \
  cargo test -p jxl-codec generated_split_vardct_exposes_global_cursor_when_available --features conformance
```

The tool expects a naked JPEG XL codestream. It uses libjxl internals for the
codestream header, frame TOC, VarDCT global preamble, and modular MA tree. The
small ANS histogram reader is mirrored locally so the tool can report per
histogram bit ranges, which libjxl does not expose through a callable helper.

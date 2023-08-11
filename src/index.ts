import * as ffi from 'ffi-napi';
import * as ref from 'ref-napi';
import * as path from 'path';
import * as fs from 'fs';

enum whisper_sampling_strategy {
  WHISPER_SAMPLING_GREEDY,      // similar to OpenAI's GreedyDecoder
  WHISPER_SAMPLING_BEAM_SEARCH, // similar to OpenAI's BeamSearchDecoder
};

const StructType = require('ref-struct-di')(ref);
const ArrayType = require('ref-array-di')(ref);

const TokenArray = ArrayType(ref.types.int);

const WhisperContext = ref.refType(ref.types.void);
const WhisperFullParams = StructType({
  strategy: ref.types.int,

  n_threads: ref.types.int,
  n_max_text_ctx: ref.types.int,     // max tokens to use from past text as prompt for the decoder
  offset_ms: ref.types.int,          // start offset in ms
  duration_ms: ref.types.int,        // audio duration to process in ms

  translate: ref.types.bool,
  no_context: ref.types.bool,        // do not use past transcription (if any) as initial prompt for the decoder
  single_segment: ref.types.bool,    // force single segment output (useful for streaming)
  print_special: ref.types.bool,     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
  print_progress: ref.types.bool,    // print progress information
  print_realtime: ref.types.bool,    // print results from within whisper.cpp (avoid it, use callback instead)
  print_timestamps: ref.types.bool,  // print timestamps for each text segment when printing realtime

  // [EXPERIMENTAL] token-level timestamps
  token_timestamps: ref.types.bool, // enable token-level timestamps
  hold_pt: ref.types.float,         // timestamp token probability threshold (~0.01)
  hold_ptsum: ref.types.float,      // timestamp token sum probability threshold (~0.01)
  max_len: ref.types.int,          // max segment length in characters
  split_on_word: ref.types.bool,    // split on word rather than on token (when used with max_len)
  max_tokens: ref.types.int,       // max tokens per segment (0 = no limit)

  // [EXPERIMENTAL] speed-up techniques
  // note: these can significantly reduce the quality of the output
  speed_up: ref.types.bool,          // speed-up the audio by 2x using Phase Vocoder
  audio_ctx: ref.types.int,         // overwrite the audio context size (0 = use default)

  // [EXPERIMENTAL] [TDRZ] tinydiarize
  tdrz_enable: ref.types.bool,       // enable tinydiarize speaker turn detection

  // tokens to provide to the whisper decoder as initial prompt
  // these are prepended to any existing text context from a previous call
  initial_prompt: ref.types.CString,
  prompt_tokens: TokenArray,
  prompt_n_tokens: ref.types.int,

  // for auto-detection, set to nullptr, "" or "auto"
  language: ref.types.CString,
  detect_language: ref.types.bool,

  // common decoding parameters:
  suppress_blank: ref.types.bool,    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
  suppress_non_speech_tokens: ref.types.bool, // ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253

  temperature: ref.types.float,      // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
  max_initial_ts: ref.types.float,   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
  length_penalty: ref.types.float,   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267

  // fallback parameters
  // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
  temperature_inc: ref.types.float,
  entropy_thold: ref.types.float,    // similar to OpenAI's "compression_ratio_threshold"
  logprob_thold: ref.types.float,
  no_speech_thold: ref.types.float,  // TODO: not implemented

  greedy: StructType({
    best_of: ref.types.int
  }),

  beam_search: StructType({
    beam_size: ref.types.int,
    patience: ref.types.float
  }),

  // called for every newly generated text segment
  new_segment_callback: ref.refType(ref.types.void),
  new_segment_callback_user_data: ref.refType(ref.types.void),

  // called on each progress update
  progress_callback: ref.refType(ref.types.void),
  progress_callback_user_data: ref.refType(ref.types.void),

  // called each time before the encoder starts
  encoder_begin_callback: ref.refType(ref.types.void),
  encoder_begin_callback_user_data: ref.refType(ref.types.void),

  // called by each decoder to filter obtained logits
  logits_filter_callback: ref.refType(ref.types.void),
  logits_filter_callback_user_data: ref.refType(ref.types.void),
});

const aprilLib = ffi.Library(
  // @ts-ignore
  new ffi.DynamicLibrary(
    './whisper.dll',
    ffi.DynamicLibrary.FLAGS.RTLD_NOW | ffi.DynamicLibrary.FLAGS.RTLD_GLOBAL
  ), {
    whisper_init_from_file: [WhisperContext, [ref.types.CString]],
    whisper_is_multilingual: [ref.types.bool, [WhisperContext]],
    whisper_free: ['void', [WhisperContext]],
    whisper_full_default_params: [WhisperFullParams, [ref.types.int]],
    whisper_reset_timings: ['void', [WhisperContext]],
    whisper_full: ['void', [WhisperContext, WhisperFullParams, ArrayType(ref.types.float), ref.types.int]],
    whisper_print_timings: ['void', [WhisperContext]],
    whisper_full_n_segments: [ref.types.int, [WhisperContext]],
    whisper_full_get_segment_text: [ref.types.CString, [WhisperContext, ref.types.int]],
    whisper_full_n_tokens: [ref.types.int, [WhisperContext, ref.types.int]],
    whisper_full_get_token_id: [ref.types.int, [WhisperContext, ref.types.int, ref.types.int]],
  }
);

const context = aprilLib.whisper_init_from_file('./ggml-base.bin');
const params = aprilLib.whisper_full_default_params(whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY);
params.print_realtime   = true;
params.print_progress   = false;
params.print_timestamps = true;
params.print_special    = false;
params.translate        = true;
params.language         = "en";
params.n_threads        = 8;
params.offset_ms        = 0;
params.single_segment   = false;
params.no_context       = false;

params.new_segment_callback = new ffi.Callback(
  'void',
  [WhisperContext, ref.refType(ref.types.void), ref.types.int, ref.refType(ref.types.void)],
  (ctx, state, n_new, user_data) => {
    console.log('new_segment_callback');
  }
);
console.log(params.prompt_tokens);

var pcm_data = fs.readFileSync('test.wav');
var pcm_data_view = new DataView(pcm_data.buffer);
var newView = new DataView(new ArrayBuffer(pcm_data.buffer.byteLength*2));

for (let i = 0; i < pcm_data.buffer.byteLength; i+=2) {
  newView.setFloat32(i*2, pcm_data_view.getInt16(i, true)/32768, true);
}

fs.writeFileSync('test2.wav', Buffer.from(newView.buffer));

var pcm_data = fs.readFileSync('test2.wav');
var pcm = new Float32Array(pcm_data.buffer);
var arr = [];
for (let i = 0; i < pcm.length; i++) {
  arr[i] = pcm[i];
}

aprilLib.whisper_reset_timings(context);
aprilLib.whisper_full(context, params, arr, arr.length);

let tokens = [];

let x = aprilLib.whisper_full_n_segments(context);
console.log(x);
for (let i = 0; i < x; i++) {
  let y = aprilLib.whisper_full_get_segment_text(context, i);
  console.log(y);
  for (let j = 0; j < aprilLib.whisper_full_n_tokens(context, i); j++) {
    let z = aprilLib.whisper_full_get_token_id(context, i, j);
    console.log(z);
    tokens.push(z);
  }
}

var pcm_data = fs.readFileSync('jfk.pcmf32');
var pcm = new Float32Array(pcm_data.buffer);
var arr = [];
for (let i = 0; i < pcm.length; i++) {
  arr[i] = pcm[i];
}

console.log('okay..');

params.prompt_tokens = tokens;
params.prompt_n_tokens = tokens.length;

console.log('okay..');

// doesn't work for some reason after setting prompt_tokens.
aprilLib.whisper_full(context, params, arr, arr.length);

x = aprilLib.whisper_full_n_segments(context);
console.log(x);
for (let i = 0; i < x; i++) {
  let y = aprilLib.whisper_full_get_segment_text(context, i);
  console.log(y);
}
console.log('test');
aprilLib.whisper_print_timings(context);
aprilLib.whisper_free(context);
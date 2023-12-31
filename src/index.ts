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

const TokenArray = ArrayType(ref.types.int32);

const GrammarElement = ArrayType(StructType({
  type: ref.types.int,
  value: ref.types.uint32,
}));

const WhisperContext = ref.refType(ref.types.void);
const WhisperFullParams = StructType({
  strategy: ref.types.int,

  n_threads: ref.types.int,
  n_max_text_ctx: ref.types.int,     // max tokens to use from past text as prompt for the decoder
  offset_ms: ref.types.int,          // start offset in ms
  duration_ms: ref.types.int,        // audio duration to process in ms

  translate: ref.types.bool,
  no_context: ref.types.bool,        // do not use past transcription (if any) as initial prompt for the decoder
  no_timestamps: ref.types.bool,     // do not generate timestamps
  single_segment: ref.types.bool,    // force single segment output (useful for streaming)
  print_special: ref.types.bool,     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
  print_progress: ref.types.bool,    // print progress information
  print_realtime: ref.types.bool,    // print results from within whisper.cpp (avoid it, use callback instead)
  print_timestamps: ref.types.bool,  // print timestamps for each text segment when printing realtime

  // [EXPERIMENTAL] token-level timestamps
  token_timestamps: ref.types.bool, // enable token-level timestamps
  thold_pt: ref.types.float,         // timestamp token probability threshold (~0.01)
  thold_ptsum: ref.types.float,      // timestamp token sum probability threshold (~0.01)
  max_len: ref.types.int,          // max segment length in characters
  split_on_word: ref.types.bool,    // split on word rather than on token (when used with max_len)
  max_tokens: ref.types.int,       // max tokens per segment (0 = no limit)

  // [EXPERIMENTAL] speed-up techniques
  // note: these can significantly reduce the quality of the output
  speed_up: ref.types.bool,          // speed-up the audio by 2x using Phase Vocoder
  debug_mode: ref.types.bool,        // enable debug_mode provides extra info (eg. Dump log_mel)
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

  // called each time before ggml computation starts
  abort_callback: ref.refType(ref.types.void),
  abort_callback_user_data: ref.refType(ref.types.void),

  // called by each decoder to filter obtained logits
  logits_filter_callback: ref.refType(ref.types.void),
  logits_filter_callback_user_data: ref.refType(ref.types.void),

  grammar_rules: GrammarElement,
  n_grammar_rules: ref.types.ulong,
  i_start_rule: ref.types.ulong,
  grammar_penalty: ref.types.float,
});


type TranscriptionSegment =  {
  text: string,
  t0: number,
  t1: number,
};

export class Whisper {
  context: ref.Pointer<void>;
  params: any;
  whisperLib: any;
  prompt_tokens: Array<number> = [];

  static readonly WHISPER_SAMPLE_RATE = 16000;
  static readonly n_samples_step      = (1e-3*3000 )*Whisper.WHISPER_SAMPLE_RATE;
  static readonly n_samples_len       = (1e-3*5000 )*Whisper.WHISPER_SAMPLE_RATE;
  static readonly n_samples_keep      = (1e-3*200  )*Whisper.WHISPER_SAMPLE_RATE;
  static readonly n_samples_30s       = (1e-3*30000)*Whisper.WHISPER_SAMPLE_RATE;

  constructor(dllPath: string, model: string) {

    this.whisperLib = ffi.Library(
      // @ts-ignore
      new ffi.DynamicLibrary(
        dllPath,
        ffi.DynamicLibrary.FLAGS.RTLD_NOW | ffi.DynamicLibrary.FLAGS.RTLD_GLOBAL
      ), {
        whisper_init_from_file: [WhisperContext, [ref.types.CString]],
        whisper_is_multilingual: [ref.types.bool, [WhisperContext]],
        whisper_free: [ref.types.void, [WhisperContext]],
        whisper_full_default_params: [WhisperFullParams, [ref.types.int]],
        whisper_reset_timings: [ref.types.void, [WhisperContext]],
        whisper_full: [ref.types.void, [WhisperContext, WhisperFullParams, ArrayType(ref.types.float), ref.types.int]],
        whisper_full_parallel: [ref.types.void, [WhisperContext, WhisperFullParams, ArrayType(ref.types.float), ref.types.int, ref.types.int]],
        whisper_print_timings: [ref.types.void, [WhisperContext]],
        whisper_full_n_segments: [ref.types.int, [WhisperContext]],
        whisper_full_get_segment_text: [ref.types.CString, [WhisperContext, ref.types.int]],
        whisper_full_get_segment_t0: [ref.types.int, [WhisperContext, ref.types.int]],
        whisper_full_get_segment_t1: [ref.types.int, [WhisperContext, ref.types.int]],
        whisper_full_n_tokens: [ref.types.int, [WhisperContext, ref.types.int]],
        whisper_full_get_token_id: [ref.types.int, [WhisperContext, ref.types.int, ref.types.int]],
      }
    );

    this.context = this.whisperLib.whisper_init_from_file(model);
    this.params = this.whisperLib.whisper_full_default_params(whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY);
    this.params.n_threads = 16;
    this.params.print_timestamps = true;
    // this.params.no_context = false;
    // this.params.n_max_text_ctx = 0;
  }

  pcm16sto32f(pcm16: Buffer): Buffer {
    var pcm_data_view = new DataView(pcm16.buffer);
    var newView = new DataView(new ArrayBuffer(pcm16.buffer.byteLength*2));

    for (let i = 0; i < pcm16.buffer.byteLength; i+=2) {
      newView.setFloat32(i*2, pcm_data_view.getInt16(i, true)/32768, true);
    }

    return Buffer.from(newView.buffer);
  }

  prepare(pcm16: Buffer) {
    var pcm_data = new Float32Array(pcm16.buffer);
    var arr = [];
    for (let i = 0; i < pcm_data.length; i++) {
      arr[i] = pcm_data[i];
    }

    return arr;
  }

  text: string = '';
  lines: string[] = [];
  pcmf32: Array<number> = [];
  pcmf32_new: Array<number> = [];

  transcribe(arr: Array<number>): string {

    let text = '';
    //this.params.prompt_tokens = this.prompt_tokens;
    //this.params.prompt_n_tokens = this.prompt_tokens.length;

    this.whisperLib.whisper_full(this.context, this.params, arr, arr.length);
    let x = this.whisperLib.whisper_full_n_segments(this.context);
    for (let i = 0; i < x; i++) {
      text += this.whisperLib.whisper_full_get_segment_text(this.context, i);

      //this.prompt_tokens = [];
      // const token_count = this.whisperLib.whisper_full_n_tokens(this.context, i);
      // for (let j = 0; j < token_count; ++j) {
      //   this.prompt_tokens.push(this.whisperLib.whisper_full_get_token_id(this.context, i, j));
      // }
    }

    // this.whisperLib.whisper_print_timings(this.context);
    // this.whisperLib.whisper_reset_timings(this.context);
    // this.whisperLib.whisper_free(this.context);

    return text;
  }

  transcribe_get_object(arr: Array<number>): TranscriptionSegment[] {

    let retArr: TranscriptionSegment[] = [];
    //this.params.prompt_tokens = this.prompt_tokens;
    //this.params.prompt_n_tokens = this.prompt_tokens.length;

    this.whisperLib.whisper_full(this.context, this.params, arr, arr.length);
    let x = this.whisperLib.whisper_full_n_segments(this.context);
    for (let i = 0; i < x; i++) {
      retArr.push({
        text: this.whisperLib.whisper_full_get_segment_text(this.context, i),
        t0: this.whisperLib.whisper_full_get_segment_t0(this.context, i),
        t1: this.whisperLib.whisper_full_get_segment_t1(this.context, i),
      });
    }

    // this.whisperLib.whisper_print_timings(this.context);
    // this.whisperLib.whisper_reset_timings(this.context);
    // this.whisperLib.whisper_free(this.context);

    return retArr;
  }

  static vad_simple(pcmf32: number[], sampleRate: number, lastMs: number, vadThold: number, freqThold: number, verbose: boolean): boolean {

    let pcmf32_old = JSON.parse(JSON.stringify(pcmf32));

    const nSamples = pcmf32_old.length;
    const nSamplesLast = Math.round(sampleRate * lastMs / 1000);

    if (nSamplesLast > nSamples) {
      // not enough samples - assume no speech

      if (verbose) {
        console.log(`nSamples: ${nSamples}, nSamplesLast: ${nSamplesLast}`);
      }
      return false;
    }

    if (freqThold > 0) {
      pcmf32_old = Whisper.highPassFilter(pcmf32_old, freqThold, sampleRate);
    }

    let energyAll = 0;
    let energyLast = 0;

    for (let i = 0; i < nSamples; i++) {
      energyAll += Math.abs(pcmf32_old[i]);

      if (i >= nSamples - nSamplesLast) {
        energyLast += Math.abs(pcmf32_old[i]);
      }
    }

    energyAll /= nSamples;
    energyLast /= nSamplesLast;

    if (verbose) {
      console.log(`energy_all: ${energyAll}, energy_last: ${energyLast}, vad_thold: ${vadThold}, freq_thold: ${freqThold}`);
    }

    return energyLast <= vadThold * energyAll;
  }

  static highPassFilter(data: number[], cutoff: number, sampleRate: number): number[] {

    const rc = 1 / (2 * Math.PI * cutoff);
    const dt = 1 / sampleRate;
    const alpha = dt / (rc + dt);

    let y = data[0];

    for (let i = 1; i < data.length; i++) {
      y = alpha * (y + data[i] - data[i - 1]);
      data[i] = y;
    }

    return data;

  }
}
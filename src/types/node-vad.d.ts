declare module 'node-vad' {

  import { Transform } from 'stream';
  import { Buffer } from 'buffer';

  class VAD {
    static Event: {
      ERROR: -1,
      SILENCE: 0,
      VOICE: 1,
      NOISE: 2
    }

    static Mode: {
      NORMAL: 0,
      LOW_BITRATE: 1,
      AGGRESSIVE: 2,
      VERY_AGGRESSIVE: 3
    }

    constructor(mode: number);
    processAudio(buffer: Buffer, rate: number): Promise<number>;
    processAudioFloat(buffer: Buffer, rate: number): Promise<number>;
    static createStream(opts: VAD.VadOptions): VAD.VADStream;
    static toFloatBuffer(buffer: Buffer): Buffer;
  }

  namespace VAD {

    export interface VadOptions {
      mode?: number;
      audioFrequency?: 8000 | 16000 | 32000 | 48000;
      debounceTime?: number;
    }

    export interface SpeechData {
      state: boolean,
      start: boolean,
      end: boolean,
      startTime: number,
      duration: number
    }

    export interface VadTransformerData {
      time: number,
      audioData: Buffer,
      speech: SpeechData
    }

    export class VADStream extends Transform {
      constructor(options?: VadOptions);

      _transform(chunk: any, encoding: string, callback: (error?: Error | null, data?: any) => void): void;
      _processAudio(chunk: Buffer): Promise<void>;
    }

  }

  export = VAD;
}

// declare module 'node-vad' {
//   // Define Enums
//   export enum Mode {
//       NORMAL,
//       LOW_BITRATE,
//       AGGRESSIVE,
//       VERY_AGGRESSIVE,
//   }

//   export enum Event {
//       ERROR,
//       NOISE,
//       SILENCE,
//       VOICE,
//   }

//   // Define methods on VAD instance
//   export interface VAD {
//       processAudio(chunk: Buffer, sampleRate: number): Promise<Event>;
//   }

//   // Define Static members on VAD
//   interface VADStatic {
//       new (mode: Mode): VAD;
//       Mode: typeof Mode;
//       Event: typeof Event;
//   }

//   // Declare VAD variable of the static type
//   const VAD: VADStatic;
//   export default VAD;
// }
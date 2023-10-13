import { Whisper } from './index';
import * as fs from 'fs';

const whisper = new Whisper('./whisper.gpu.dll', './ggml-tiny.bin');
const wav = fs.readFileSync('sample2.wav');
const len = wav.length;
const chunkLen = Math.round(len / 60);

let rolling: Array<number> = [];
for (let i = 0; i < len; i += chunkLen) {
  const chunk = Buffer.from(wav.slice(i, i + chunkLen));

  //console.log(i, i + chunkLen, chunk);
  const pcm = whisper.pcm16sto32f(chunk);
  const arr = whisper.prepare(pcm);

  rolling = rolling.concat(arr);

  if (Whisper.vad_simple(rolling, Whisper.WHISPER_SAMPLE_RATE, 100, 0.8, 100.0, false)) {
    const ret = whisper.transcribe(rolling);
    console.log('--' + ret);
  }
}

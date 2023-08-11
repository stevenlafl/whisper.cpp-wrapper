import { Whisper } from './index';
import * as fs from 'fs';

const whisper = new Whisper('./ggml-base.bin');
const wav = fs.readFileSync('sample2.wav');
const len = wav.length;
const chunkLen = Math.round(len / 60);
for (let i = 0; i < len; i += chunkLen) {
  const chunk = Buffer.from(wav.slice(i, i + chunkLen));

  //console.log(i, i + chunkLen, chunk);
  const pcm = whisper.pcm16sto32f(chunk);
  const arr = whisper.prepare(pcm);
  const ret = whisper.transcribe(arr);
  console.log(ret);
}

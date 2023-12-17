import { Whisper } from './index';
import * as fs from 'fs';

import VAD from 'node-vad';
import Speaker from 'speaker';

const whisper = new Whisper('./whisper.gpu.cublas12.dll', './ggml-tiny.bin');

const wav = fs.readFileSync('sample2.wav');
const len = wav.length;
const chunkLen = Math.round(len / 60);

let rolling: Buffer = Buffer.alloc(0);

let speaker = new Speaker({
  channels: 1,
  bitDepth: 16,
  sampleRate: 16000,
});

(async () => {

  // VAD processing
  const vad = VAD.createStream({
    mode: VAD.Mode.NORMAL, // VAD mode, see above
    audioFrequency: 16000, // Audiofrequency, see above
    debounceTime: 1000 // Time for debouncing speech active state, default 1 second
  });

  vad.on("data", async (state) => {
    // state.audioData is non-overlapping buffer of audio data.
    rolling = Buffer.concat([rolling, state.audioData]);
    
    // If end of speech
    if (state.speech.end) {

      const pcm = whisper.pcm16sto32f(rolling);
      const arr = whisper.prepare(pcm);
      const res = whisper.transcribe_get_object(arr);
      console.log(JSON.stringify(res, null, 2) + '\n');
      //console.log(state);

      rolling = Buffer.alloc(0);
    }
    //console.log(state);
    //speaker.write(state.audioData);
  });

  for (let s = 0; s < 10; s += 1) {
    for (let i = 0; i < len; i += chunkLen) {
      const chunk = Buffer.from(wav.slice(i, i + chunkLen));

      // const pcm = whisper.pcm16sto32f(chunk);
      // const arr = whisper.prepare(pcm);
      // const res = whisper.transcribe(arr);
      // console.log(res);

      await vad._processAudio(chunk);

      // await (new Promise((resolve) => {
      //   setTimeout(resolve, (chunk.length / 2 / 16000 * 1000));
      // }));
    }
  }

  console.log('Done');
})();

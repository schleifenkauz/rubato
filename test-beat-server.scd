NetAddr.langPort

OSCdef.trace(false, true)

s.boot

(
SynthDef(\click) {
	var snd = Impulse.ar(0);
	snd = BPF.ar(snd, 2000, 0.1) * 50 * \amp.kr(1);
	snd = Pan2.ar(snd);
	snd = snd * Env.perc(0, 0.1).kr(Done.freeSelf);
	Out.ar(0, snd);
}.add;
)

Synth(\click)

(
OSCdef(\beat, { |msg|
	var data = msg[1..], magn;
	magn = data[0];
	"beat".postln;
	Synth(\click, [amp: 5])
}, \beat);
)
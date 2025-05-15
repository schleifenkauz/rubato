NetAddr.langPort

OSCdef.trace(false, true)

s.reboot

(
SynthDef(\click) {
	var snd = Impulse.ar(0);
	snd = BPF.ar(snd, 2000, 0.1) * 50 * \amp.kr(1);
	snd = Pan2.ar(snd);
	snd = snd * Env.perc(0, 0.1).kr(Done.freeSelf);
	Out.ar(0, snd);
}.add;

SynthDef(\vst_instr) {
	var snd = VSTPlugin.ar(nil, 2);
	Out.ar(0, snd);
}.add;

SynthDef(\blip) {
	var snd = SinOsc.ar(\freq.kr(440));
	snd = snd * Env.perc(0.02, 0.2, \amp.kr(0.1)).kr(Done.freeSelf);
	snd = Pan2.ar(snd, \pan.kr(0));
	Out.ar(0, snd);
}.add;
)

Synth(\blip)

~pitch = 60;

(
OSCdef(\beat, { |msg|
	var data = msg[1..], amp;
	amp = data[0].linlin(20, 200, 0.1, 1);
	"beat".postln;
	Synth(\blip, [amp: amp, freq: ~pitch.midicps])
}, \beat);

OSCdef(\left_hand, { |msg|
	var data = msg[1..], x, y, fx;
	//data.postln;
	if (data.size == 2) {
		x = data[0];
		~pan = x.linlin(0, 1, -1, 1);
		y = data[1];
		~pitch = y.linlin(0, 1, 40, 80);
	}
}, \left_hand);
)
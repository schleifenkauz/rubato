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
}.add
)

VSTPlugin.search

(
fork {
	x.free;
	x = Synth(\vst_instr);
	s.sync;
	~labs = VSTPluginController(x).open('LABS');
	s.sync;
	~labs.editor;
}
)

(fork {
	for(0, 48) { |i|
		i.postln;
		~labs.set(i, 1);
		0.5.wait;
	}
})

Synth(\click)

~labs.midi.noteOn(0, 50)

~pitch = 60;

(
OSCdef(\beat, { |msg|
	var data = msg[1..], magn;
	magn = data[0];
	"beat".postln;
	fork {
		~labs.midi.noteOn(0, ~pitch, magn.linlin(50, 200, 10, 127));
		0.3.wait;
		~labs.midi.noteOff(0, 60)
	}
	//Synth(\click, [amp: 5])
}, \beat);
)

(
OSCdef(\left_hand, { |msg|
	var data = msg[1..], x, y, fx;
	data.postln;
	if (data.size == 2) {
		y = data[1];
		~pitch = y.linlin(0, 1, 40, 80).postln;
		x = data[0];
		fx = x.linlin(0, 0.5, 0, 1).postln;
		~labs.set(17, fx);
	}
}, \left_hand);
)
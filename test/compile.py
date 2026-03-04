@DAG
def tone(freq=440):
    Out.ar(0, Pan(SinOsc.ar(freq)))

dsp.send(Compile(tone))
dsp.sync()

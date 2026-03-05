from mc1.dag import DAG, Out, Pan, SinOsc


@DAG
def tone(freq=440, amp=0.2):
    Out.ar(0, Pan(SinOsc.ar(freq) * amp))

lines = []
step = 0.5
m = 0
s = 0
ms = 0
for i in range(int((28 * 60 + 6) / step)):
    ms += step
    if ms >= 1:
        ms -= 1
        s += 1
    if s >= 60:
        s -= 60
        m += 1

    lines.append("0-" + str(m) + ":" + str(s) + ":" + str(ms))

thefile = open('/home/sebi/audio-tests/tagged.txt', 'w')
for item in lines:
  thefile.write("%s\n" % item)
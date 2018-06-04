def create_tag_file():
    lines = []
    step = 5000
    h = 0
    m = 0
    s = 0
    ms = 0
    for i in range(0, 120 * 60 * 1000, step):
        ms += step
        if ms >= 1000:
            ms -= step
            s += int(step/1000)
        if s >= 60:
            s -= 60
            m += 1

        if m >= 60:
            m -= 60
            h += 1

        lines.append("0-" + '{num:02d}'.format(num=h) + ":" + '{num:02d}'.format(num=m) + ":" + '{num:02d}'.format(num=s) + ":" + '{num:02d}'.format(num=ms))

    thefile = open('tagged_empty.txt', 'w')
    for item in lines:
      thefile.write("%s\n" % item)


def change_tag_interval():
    steps = int(10 * 0.5)

    input_file = open('tagged.txt', 'r')
    out = open('tagged_new.txt', 'w')

    tagged = [x.strip() for x in input_file.readlines()]
    new_lines = []
    for i in range(0, len(tagged), steps):
        tag = 0
        if 1 in [int(t[0]) for t in tagged[i:i+steps]]:
            tag = 1

        new_lines.append(tag)

    for item in new_lines:
      out.write("%s\n" % item)

create_tag_file()
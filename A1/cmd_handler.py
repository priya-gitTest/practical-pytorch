class CMDHandler:
    def run(self, in_stream):
        cmds = in_stream.read()
        idx = 0
        while idx < len(cmds):
            if cmds[idx] == 'o':
                idx += 1
                print(self.add_to_history(cmds[idx]))
            elif cmds[idx] == 'q':
                idx += 1
                print('%f' % self.log_prob(cmds[idx]))
            elif cmds[idx] == 'g':
                c = self.generate()
                print(c)
                self.add_to_history(c)
            elif cmds[idx] == 'x':
                break
            else:
                raise Exception('invalid cmd')
            idx += 1

    def add_to_history(self, c):
        return 'add to history %s' % c

    def log_prob(self, c):
        return 1

    def generate(self):
        return 'g'


if __name__ == '__main__':
    import sys
    import numpy as np

    seed = int(sys.argv[1])
    np.random.seed(seed)

    cmd_handler = CMDHandler()
    cmd_handler.run(sys.stdin)



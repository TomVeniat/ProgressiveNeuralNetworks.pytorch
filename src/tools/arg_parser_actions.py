import argparse


class LengthCheckAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != namespace.layers:
            msg = "Sizes must have length L (number of layers). L={}, got {} values"
            parser.error(msg.format(namespace.layers, len(values)))

        setattr(namespace, self.dest, values)

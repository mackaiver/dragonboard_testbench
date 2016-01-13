""" Plot Dragon Board Raw

Show drag data interactively. Just click on the plotting canvas to
see the next event.


Usage:
  plot_drago_data.py <filename>
  plot_drago_data.py (-h | --help)
  plot_drago_data.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

"""
from docopt import docopt
from dragonboard import EventGenerator, DragonBrowser


def main():
    arguments = docopt(__doc__, version='Dragon Data Browser 0.1alpha')

    with open(arguments["<filename>"], "rb") as f:
        generator = EventGenerator(f)

        browser = DragonBrowser(generator)
        browser.fig.suptitle(
            "DragoCam raw data. {}".format(arguments["<filename>"])
        )
        browser.show()

if __name__ == '__main__':
    main()

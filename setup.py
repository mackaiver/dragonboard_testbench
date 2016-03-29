from setuptools import setup

setup(
    name='dragonboard',
    version='0.0.2',
    description='A reader library for the cta dragonboard data',
    url='http://github.com/cta-observatory/dragonboard_testbench',
    author='Kai Brügge, Mario Hörbe, Dominik Neise, Maximilian Nöthe',
    author_email='kai.bruegge@tu-dortmund.de',
    license='MIT',
    packages=['dragonboard'],
    entry_points={
        'gui_scripts': [
            'dragonviewer = dragonboard.__main__:main'
        ]
    }
)

from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tactile_contact_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iker_casti',
    maintainer_email='iker_casti@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'contact_detector_node = tactile_contact_detector.detector_node:main'
        ],
    },
)

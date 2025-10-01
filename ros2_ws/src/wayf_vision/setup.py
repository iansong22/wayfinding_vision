from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wayf_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='isong',
    maintainer_email='iansong2@illinois.edu',
    description='Package to run nodes necessary for wayfinding people tracking',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'yolo_node = wayf_vision.yolo_node:main',
            'drspaam_node = wayf_vision.drspaam_node:main',
            'kalman_node = wayf_vision.kalman_node:main',
        ],
    },
)

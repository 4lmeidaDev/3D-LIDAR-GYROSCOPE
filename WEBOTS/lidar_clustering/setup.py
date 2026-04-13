import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lidar_clustering'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='TODO@email.com',
    description='ROS2 DBSCAN clustering node para LiDAR point clouds (Webots)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'clustering_node = {package_name}.clustering_node:main',
        ],
    },
)

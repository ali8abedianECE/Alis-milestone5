from setuptools import setup

package_name = 'alia'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali8abedian',
    maintainer_email='mabedi02@student.ubc.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    alias_require=['pyalia'],
    entry_points={
        'console_scripts': [
            'data_fuser_node = alia.data_fuser_node:main',
            'car_detector_node = alia.car_detector_node:main',
            'normal_node = alia.normal_node:main',
            'car_follow_node = alia.car_follow_node:main',
            'car_follow_advanced_node = alia.car_follow_advanced_node:main',
            'car_overtake_node = alia.car_overtake_node:main',
            'car_overtake_slam_node = alia.car_overtake_slam_node:main',
            'raceline_recorder_node = alia.raceline_recorder_node:main',
            'slam_data_collector_node = alia.slam_data_collector_node:main',
            'arbiter_node = alia.arbiter_node:main',
            'camera_aeb_node = alia.camera_aeb_node:main',
        ],
    },
)

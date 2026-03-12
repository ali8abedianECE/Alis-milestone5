from setuptools import setup

package_name = 'milestone5'

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
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_fuser_node = milestone5.data_fuser_node:main',
            'car_detector_node = milestone5.car_detector_node:main',
            'normal_node = milestone5.normal_node:main',
            'car_follow_node = milestone5.car_follow_node:main',
            'car_follow_advanced_node = milestone5.car_follow_advanced_node:main',
            'car_overtake_node = milestone5.car_overtake_node:main',
            'car_overtake_slam_node = milestone5.car_overtake_slam_node:main',
            'raceline_recorder_node = milestone5.raceline_recorder_node:main',
            'slam_data_collector_node = milestone5.slam_data_collector_node:main',
            'arbiter_node = milestone5.arbiter_node:main',
            'camera_aeb_node = milestone5.camera_aeb_node:main',
        ],
    },
)

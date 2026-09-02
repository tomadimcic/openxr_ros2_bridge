from setuptools import find_packages, setup

package_name = 'openxr_ros2_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tomislav Dimcic',
    maintainer_email='toma.dimcic@gmail.com',
    description='ARISE Tele-Craft Module A: OpenXR poses to TwistStamped',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'openxr_ros2_bridge_node = openxr_ros2_bridge.openxr_ros2_bridge_node:main'
        ],
    },
)

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crearobot_brain'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marco',
    maintainer_email='marco.grandclaude@etu.sorbonne-universite.fr',
    entry_points={
        'console_scripts': [
            'orchestrator = crearobot_brain.orchestrator_node:main',
            'interaction = crearobot_brain.interaction_node:main',
            'brain = crearobot_brain.brain_node:main',
            'gateway = crearobot_brain.gateway_node:main',
            'projection = crearobot_brain.projection_node:main',
            'vision = crearobot_brain.vision_node:main',
            'stt = crearobot_brain.stt_node:main',
        ],
    },
)

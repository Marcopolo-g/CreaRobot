from setuptools import find_packages, setup

package_name = 'crearobot_brain'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marco',
    maintainer_email='marco.grandclaude@etu.sorbonne-universite.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'orchestrator = crearobot_brain.orchestrator_node:main',
            'interaction = crearobot_brain.interaction_node:main',
            'llm_reactif_node = crearobot_brain.llm_reactif_node:main',
            'gateway = crearobot_brain.gateway_node:main',
            'vision = crearobot_brain.vision_node:main',
            'gateway_test = crearobot_brain.tests_finaux.gateway_node_test:main',
            'brain_test = crearobot_brain.tests_finaux.brain_node_test:main',
            'camera_test = crearobot_brain.tests_finaux.camera_node_test:main',
            'stt = crearobot_brain.stt_node:main',
        ],
    },
)

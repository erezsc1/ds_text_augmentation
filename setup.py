import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AugmenText",
    version="0.0.1",
    author="erezsc",
    author_email="erezsc@rnd-hub.com",
    description="Text Augmentation library based on MarianMT translator service ('Miriam')",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    py_modules=["text_augmentaion"],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "requests ~= 2.18",
        "translator_client ~= 0.0.2"
    ]
)

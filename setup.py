#!/usr/bin/env python
"""setup script for building, distributing and installing."""

from setuptools import setup, find_packages


with open("README.md") as readme, open("CHANGELOG.md") as changelog:
    LONG_DESCRIPTION = readme.read() + 2 * '\n' + changelog.read()

setup(
    name='neudataload',
    description="Neurological Data Loading",
    long_description=LONG_DESCRIPTION,
    author="Efraín Lima Miranda",
    author_email='efrain70@gmail.com',
    url='https://github.com/efrain70/neudataload',
    packages=find_packages(),
    include_package_data=True,
    use_scm_version={'write_to': 'neudataload/version.py'},
    setup_requires=[
        'setuptools_scm',
    ],
    install_requires=[
        'pandas',
        'xlrd',
    ],
    license='CC BY-NC-SA 4.0',
    zip_safe=False,
    keywords=['neudataload'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)

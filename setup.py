import setuptools
with open("README.md","r",encoding="utf-8") as f:
    long_description=f.read()

    __version__="0.0.0"
    REPO_NAME="mlProject"
    AUTHOR_USER_NAME="axk173030"
    SRC_REPO="mlProject"

    setuptools.setup(
        name=SRC_REPO,
        version=__version__,
        author=AUTHOR_USER_NAME,
        description="A small python package for mlops",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
        project_urls={
            "BUg Tracker":f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
            
        },
        package_dir={"":"src"},
        packages=setuptools.find_packages(where="src")
    )
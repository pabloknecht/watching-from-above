reinstall_package:
	@pip uninstall -y wfa || :
	@pip install -e .

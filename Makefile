reinstall_package:
	@pip uninstall -y wfa || :
	@pip install -e .

run_api:
	uvicorn wfa.api.fast:app --reload

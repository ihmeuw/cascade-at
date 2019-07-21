from cascade.runner.application_config import application_config


def test_parameter_file_proper_format():
    """Confirms sections found in the ini file."""
    config = application_config()
    for section in ["DataLayout", "Database"]:
        assert config.has_section(section)


def test_read_parameters():
    """
    Reads a couple of parameters.
    """
    config = application_config()
    assert isinstance(config["DataLayout"]["epiviz-log-directory"], str)
    assert isinstance(config["Database"]["corporate-odbc"], str)

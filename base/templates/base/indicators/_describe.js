get_value_funs["describe"] = function get_current_quaartile_value(list_item) {
	text = list_item.getElementsByClassName("current-quartile")[0].textContent;
	return new Number(text)
};

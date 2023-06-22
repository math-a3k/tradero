get_value_funs["ac"] = function get_ac_value(list_item) {
	text = list_item.getElementsByClassName("ac-value")[0].textContent;
	return new Number(text.substring(0, text.length - 1))
};

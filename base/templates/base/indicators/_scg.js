get_value_funs["scg"] = function get_ac_value(list_item) {
	text = list_item.getElementsByClassName("scg-value")[0].textContent;
	return new Number(text)
};

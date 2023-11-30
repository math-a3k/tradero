get_value_funs["atr"] = function get_atr_value(list_item) {
	text = list_item.getElementsByClassName("atr-value")[0].textContent;
	return new Number(text)
};

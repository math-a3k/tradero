get_value_funs["dc"] = function get_dc_value(list_item) {
	text = list_item.getElementsByClassName("dc-value")[0].innerHTML;
	return text.includes("YES")
};

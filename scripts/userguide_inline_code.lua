local function latex_escape(value)
  local replacements = {
    ["\\"] = "\\textbackslash{}",
    ["{"] = "\\{",
    ["}"] = "\\}",
    ["$"] = "\\$",
    ["&"] = "\\&",
    ["#"] = "\\#",
    ["%"] = "\\%",
    ["_"] = "\\_",
    ["^"] = "\\textasciicircum{}",
    ["~"] = "\\textasciitilde{}",
  }
  return (value:gsub("[\\{}$&#%%_^~]", replacements))
end

function Code(element)
  return pandoc.RawInline("latex", "\\texttt{" .. latex_escape(element.text) .. "}")
end

local function latex_escape(value)
  local replacements = {
    ["\\"] = "\\textbackslash{}",
    ["{"] = "\\{",
    ["}"] = "\\}",
    ["$"] = "\\$",
    ["&"] = "\\&",
    ["#"] = "\\#",
    ["%"] = "\\%",
    ["_"] = "\\_\\allowbreak{}",
    ["^"] = "\\textasciicircum{}",
    ["~"] = "\\textasciitilde{}",
  }
  value = value:gsub("[\\{}$&#%%_^~]", replacements)
  value = value:gsub("/", "/\\allowbreak{}")
  value = value:gsub("%-", "-\\allowbreak{}")
  value = value:gsub("%.", ".\\allowbreak{}")
  return value
end

function Code(element)
  return pandoc.RawInline("latex", "\\texttt{" .. latex_escape(element.text) .. "}")
end

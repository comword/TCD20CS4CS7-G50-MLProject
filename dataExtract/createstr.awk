function parse_csv(string,csv,sep,quote,escape,newline,trim, fields,pos,strtrim) {
    # Make sure there is something to parse.
    if (length(string) == 0) return 0;
    string = sep string; # The code below assumes ,FIELD.
    fields = 0; # The number of fields found thus far.
    while (length(string) > 0) {
        # Remove spaces after the separator if requested.
        if (trim && substr(string, 2, 1) == " ") {
            if (length(string) == 1) return fields;
            string = substr(string, 2);
            continue;
        }
        strtrim = 0; # Used to trim quotes off strings.
        # Handle a quoted field.
        if (substr(string, 2, 1) == quote) {
            pos = 2;
            do {
                pos++
                if (pos != length(string) &&
                    substr(string, pos, 1) == escape &&
                    (substr(string, pos + 1, 1) == quote ||
                     substr(string, pos + 1, 1) == escape)) {
                    # Remove escaped quote characters.
                    string = substr(string, 1, pos - 1) substr(string, pos + 1);
                } else if (substr(string, pos, 1) == quote) {
                    # Found the end of the string.
                    strtrim = 1;
                } else if (newline && pos >= length(string)) {
                    # Handle embedded newlines if requested.
                    if (getline == -1) {
                        csverr = "Unable to read the next line.";
                        return -1;
                    }
                    string = string newline $0;
                }
            } while (pos < length(string) && strtrim == 0)
            if (strtrim == 0) {
                csverr = "Missing end quote.";
                return -2;
            }
        } else {
            # Handle an empty field.
            if (length(string) == 1 || substr(string, 2, 1) == sep) {
                csv[fields] = "";
                fields++;
                if (length(string) == 1)
                    return fields;
                string = substr(string, 2);
                continue;
            }
            # Search for a separator.
            pos = index(substr(string, 2), sep);
            # If there is no separator the rest of the string is a field.
            if (pos == 0) {
                csv[fields] = substr(string, 2);
                fields++;
                return fields;
            }
        }
        # Remove spaces after the separator if requested.
        if (trim && pos != length(string) && substr(string, pos + strtrim, 1) == " ") {
            trim = strtrim
            # Count the number fo spaces found.
            while (pos < length(string) && substr(string, pos + trim, 1) == " ") {
                trim++
            }
            # Remove them from the string.
            string = substr(string, 1, pos + strtrim - 1) substr(string,  pos + trim);
            # Adjust pos with the trimmed spaces if a quotes string was not found.
            if (!strtrim) {
                pos -= trim;
            }
        }
        # Make sure we are at the end of the string or there is a separator.
        if ((pos != length(string) && substr(string, pos + 1, 1) != sep)) {
            csverr = "Missing separator.";
            return -3;
        }
        # Gather the field.
        csv[fields] = substr(string, 2 + strtrim, pos - (1 + strtrim * 2));
        fields++;
        # Remove the field from the string for the next pass.
        string = substr(string, pos + 1);
    }
    return fields;
}


{

  if(FNR==1){
    num_fields = parse_csv($0, csv, ",", "\"", "\"", "\\n", 1);
    if (num_fields < 0) {
        printf "ERROR: %s (%d) -> %s\n", csverr, num_fields, $0;
    }
  }else{
    num_fields = parse_csv($0, csvw, ",", "\"", "\"", "\\n", 1);
    if (num_fields < 0) {
        printf "ERROR: %s (%d) -> %s\n", csverr, num_fields, $0;
    }

    icode = csvw[3]

    KnownType[icode]=1
  }
  if(FNR>=10000){
    nextfile
  }
}
END{
  BEGINNING="CREATE TABLE \"data\" ( \"CountryCode\"	TEXT NOT NULL, \"Year\"	INTEGER NOT NULL,"

  #"A.b"	REAL NOT NULL,

  MIDDLE = ""

  ENDDING=" UNIQUE(\"CountryCode\",\"Year\"));"
  for(k in KnownType){
    MIDDLE = MIDDLE "\"" k "\" REAL,"
  }

  print BEGINNING MIDDLE ENDDING
}

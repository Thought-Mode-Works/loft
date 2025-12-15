"Fix: results["missing_rules"] is already a List[str]
                # It should use .extend() if appending multiple items from an iterable
                # or append each item individually."
                results["missing_rules"].extend(list(missing)[:5])
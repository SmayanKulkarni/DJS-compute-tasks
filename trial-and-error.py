common_districts = []

for district in your_districts:
    if district in their_districts and district not in common_districts:
        common_districts.append(district)

for district in common_districts:
  if district in patrolled_districts:
    common_districts.remove(district)

print(common_districts)
  
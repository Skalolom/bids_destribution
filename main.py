import numpy as np
from optimization import get_quantity_difference, distribute_bids, create_report, get_energy_proposal,\
	get_quantity_difference_new, calculate_quantities_diff, equalize_proposal_matrices

if __name__ == '__main__':
	first_proposal_matrix = np.array([[1, 2, 2.5, 3, 4, 4.5, 5], [10, 20, 25, 30, 40, 45, 50]], dtype=float)
	second_proposal_matrix = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=float)
	base_proposal_matrix, adjusted_proposal_matrix = equalize_proposal_matrices(
		proposal_matrices=(first_proposal_matrix, second_proposal_matrix)
	)
	# quantities_start = np.array(
	# 	[
	# 		2266.663, 2516.663, 2761.663, 3001.663, 3081.663, 3441.663, 3526.663, 3551.673, 3606.683, 3846.683, 4086.683,
	# 		4176.683, 4416.683, 4646.683, 4846.683, 4956.703, 5196.703, 5436.703, 5636.203, 5746.203, 5806.203, 5905.213,
	# 		5995.213, 5995.223, 6075.223, 6135.233, 6224.233, 6289.233, 6354.233, 6384.233, 6419.243, 6479.243, 6539.243,
	# 		8083.743
	# 	]
	# )
	# quantities_finish = np.array(
	# 	[
	# 		2624.777, 2874.777, 3119.777, 3359.777, 3439.777, 3799.777, 3884.777, 3909.787, 3964.797, 4204.797, 4444.797,
	# 		4534.797, 4774.797, 5004.797, 5204.797, 5314.817, 5554.817, 5794.817, 5994.317, 6104.317, 6164.317, 6263.327,
	# 		6353.327, 6353.337, 6433.337, 6493.347, 6582.347, 6647.347, 6712.347, 6742.347, 6777.357, 6837.357, 6897.357,
	# 		9152.857
	# 	]
	# )
	# prices = np.array(
	# 	[
	# 		1631, 1756, 1821, 1823, 1849, 1859, 1891, 1893, 1894, 1898, 1920, 1926, 1944, 1946, 2000, 2004, 2037, 2056,
	# 		2100, 2109, 2116, 2189, 2221, 2223, 2275, 2326, 2395, 2413, 2429, 2619, 2981, 3895, 4099, 20000
	# 	]
	# )
	previous_hour_filename = 'bids_distribution_prev.csv'
	energy_proposal_in_current_hour = get_energy_proposal()
	energy_proposal_in_previous_hour = get_energy_proposal(filename=previous_hour_filename)

	# here we get the width of stairs in all possible proposal volumes
	quantities_all_in_current_hour = energy_proposal_in_current_hour.loc[
		energy_proposal_in_current_hour['type'] == 0, 'value'
	].to_numpy()
	delta_quantities_all_in_current_hour = np.diff(quantities_all_in_current_hour, prepend=0)
	quantities_selected_in_current_hour = energy_proposal_in_current_hour.loc[
		energy_proposal_in_current_hour['type'] == 1, 'value'
	].to_numpy()
	delta_quantities_selected_in_current_hour = np.diff(quantities_selected_in_current_hour, prepend=0)
	# quantities_all_in_current_hour = np.insert(
	# 	quantities_all_in_current_hour,
	# 	2,
	# 	quantities_all_in_current_hour[2]
	# )
	# quantities_selected_in_current_hour = np.insert(
	# 	quantities_selected_in_current_hour,
	# 	2,
	# 	quantities_selected_in_current_hour[2]
	# )
	unselected_quantities_in_current_hour = get_quantity_difference(
		quantities_start=quantities_selected_in_current_hour,
		quantities_finish=quantities_all_in_current_hour
	)

	# here we get the width of stairs in all possible proposal volumes
	quantities_all_in_previous_hour = energy_proposal_in_previous_hour.loc[
		energy_proposal_in_previous_hour['type'] == 0, 'value'
	].to_numpy()
	delta_quantities_all_in_previous_hour = np.diff(quantities_all_in_previous_hour, prepend=0)
	quantities_selected_in_previous_hour = energy_proposal_in_previous_hour.loc[
		energy_proposal_in_previous_hour['type'] == 1, 'value'
	].to_numpy()
	delta_quantities_selected_in_previous_hour = np.diff(quantities_selected_in_previous_hour, prepend=0)
	unselected_quantities_in_previous_hour = get_quantity_difference(
		quantities_start=quantities_selected_in_previous_hour,
		quantities_finish=quantities_all_in_previous_hour
	)

	stopped_volumes = get_quantity_difference(
		quantities_finish=unselected_quantities_in_previous_hour,
		quantities_start=unselected_quantities_in_current_hour
	)

	previous_prices = energy_proposal_in_previous_hour.loc[
		energy_proposal_in_previous_hour['type'] == 0, 'price'
	]
	current_prices = energy_proposal_in_current_hour.loc[
		energy_proposal_in_current_hour['type'] == 0, 'price'
	]
	res = np.invert(np.isin(previous_prices, current_prices))
	non_existed_quantity = quantities_all_in_previous_hour[res]
	down_volumes = np.array([200, 221])
	# delta_quantities = get_quantity_difference(
	# 	quantities_start=quantities_start,
	# 	quantities_finish=quantities_finish
	# )
	# bids_distribution = distribute_bids(
	# 	delta_quantities=delta_quantities,
	# 	prices_vector=prices,
	# 	up_volumes=down_volumes
	# )
	# create_report(
	# 	info_line='Московская область, 14.09, 2 останова в 16 ч. на 421 МВт (200/221)',
	# 	quantities_start=quantities_start,
	# 	quantities_finish=quantities_finish,
	# 	bids_distribution=bids_distribution,
	# 	prices=prices
	# )

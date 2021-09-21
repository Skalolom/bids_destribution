import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Tuple, Union, Any


def optimize_test(
		quantities_start: np.ndarray,
		quantities_finish: np.ndarray,
		prices_start: np.ndarray,
		prices_finish: np.ndarray,
		quantity: float
):
	delta_q_1 = np.diff(quantities_start, prepend=0)
	delta_q_2 = np.diff(quantities_finish, prepend=0)
	delta_q = delta_q_2 - delta_q_1
	prices = prices_finish
	n = prices.size
	residual = quantity - np.sum(delta_q)
	# A = np.vstack(
	# 	(np.ones(n), -1 * np.ones(n), -1 * delta_q)
	# )
	A = np.append(-1 * delta_q, -quantity)
	# b = np.array([3, -1, -1*(quantity-residual)]).reshape(-1)
	b = np.array([-1 * quantity])
	# x = cp.Variable(n, boolean=[(n,)])
	x = cp.Variable(n+1, nonneg=True)
	c = np.append(delta_q * prices, 1e8)
	prob = cp.Problem(
		cp.Minimize(c @ x),
		[A @ x <= b, x <= 1],
	)
	prob.solve(verbose=True)

	return x.value


def get_quantity_difference(
		quantities_start: np.ndarray,
		quantities_finish: np.ndarray
) -> np.ndarray:
	"""
	эта функция вычисляет изменение ширины ступеней по объему
	:param quantities_start: начальные значения объемов ступеней
	:param quantities_finish: финишные значения объемов ступеней
	:return: вектор с изменениями ширины ступеней
	"""
	n_diff = quantities_finish.size - quantities_start.size
	delta_quantities_start = np.array([])
	if n_diff > 0:
		if quantities_finish[0] < quantities_start[0]:
			delta_quantities_start = np.append(
				np.zeros(n_diff),
				np.diff(quantities_start, prepend=0)
			)
		else:
			delta_quantities_start = np.append(
				np.diff(quantities_start, prepend=0),
				np.zeros(n_diff)
			)
	else:
		delta_quantities_start = np.diff(quantities_start, prepend=0)
	delta_quantities_finish = np.diff(
		quantities_finish, prepend=0
	)
	delta_quantities = delta_quantities_finish - delta_quantities_start
	return delta_quantities


def get_quantity_difference_new(
		quantities_start: np.ndarray,
		quantities_finish: np.ndarray
) -> np.ndarray:
	"""
	эта функция вычисляет изменение ширины ступеней по объему
	:param quantities_start: начальные значения объемов ступеней
	:param quantities_finish: финишные значения объемов ступеней
	:return: вектор с изменениями ширины ступеней
	"""
	delta_quantities_start = np.diff(quantities_start, prepend=0)
	delta_quantities_finish = np.diff(quantities_finish, prepend=0)
	delta_quantities = calculate_quantities_diff(
		delta_quantities_short=delta_quantities_start,
		delta_quantities_long=delta_quantities_finish
	)
	return delta_quantities


def calculate_quantities_diff(
		delta_quantities_short: np.ndarray,
		delta_quantities_long: np.ndarray
) -> np.ndarray:
	"""
	this function calculates the difference in 2 quantities vectors
	:param delta_quantities_short: vector 1
	:param delta_quantities_long: vector 2
	:return: vector of differences
	"""
	delta_quantities = np.zeros_like(delta_quantities_long)
	for element in delta_quantities_short:
		iteration_delta = np.abs(delta_quantities_long - element)
		ind_min = np.argmin(iteration_delta)
		delta_quantities[ind_min] = element
	return delta_quantities


def distribute_bids(
		delta_quantities: np.ndarray,
		prices_vector: np.ndarray,
		up_volumes: np.ndarray
) -> np.ndarray:
	"""
	Эта функция распределяет пуски, объемы которых указываются в векторе up_volumes, по ценовым ступеням,
	которые определяются вектором delta_quantities.

	:param delta_quantities: изменение объемов ступеней
	:param prices_vector: соответствующие цены ступеней
	:param up_volumes: объемы запланированных пусков
	:return: распределение объемов пусков по ступеням
	"""
	num_vars = delta_quantities.size
	num_ups = up_volumes.size
	x = cp.Variable(num_ups * (num_vars + 1), nonneg=True)
	A = np.array([])
	for pos in range(num_vars):
		cur_vect = np.zeros(num_vars)
		cur_vect[pos] = 1
		cur_vect = np.tile(cur_vect, num_ups)
		cur_vect = np.append(
			cur_vect, np.zeros(num_ups)
		)
		if not pos:
			A = cur_vect
		else:
			A = np.vstack((A, cur_vect))
	for pos in range(num_ups):
		cur_vect = np.zeros(num_ups*(num_vars+1))
		cur_vect[pos*num_vars:(pos+1)*num_vars] = -1 * delta_quantities
		cur_vect[-2+pos] = -up_volumes[pos]
		A = np.vstack((A, cur_vect))
	b = np.append(np.ones(num_vars), -1*up_volumes)
	c = np.append(
		np.tile(delta_quantities * prices_vector, num_ups),
		1e8*np.ones(num_ups)
	)
	prob = cp.Problem(
		cp.Minimize(c @ x),
		[A @ x <= b],
	)
	prob.solve(verbose=True)
	bids_distribution = x.value * np.append(np.tile(delta_quantities, num_ups), up_volumes)
	undistributed_volumes = bids_distribution[-2:]
	bids_distribution = np.reshape(bids_distribution[:-2], (num_ups, num_vars))
	bids_distribution = np.hstack((bids_distribution, undistributed_volumes.reshape(-1, 1)))
	return bids_distribution


def create_report(
		info_line: str,
		quantities_start: np.ndarray,
		quantities_finish: np.ndarray,
		bids_distribution: np.ndarray,
		prices: np.ndarray
) -> None:
	"""
	эта функция формирует отчет по распределению объема пуска/останова по ступеням
	:param info_line: строка с описанием конкретного кейса
	:param quantities_start: исходная кривая предложения
	:param quantities_finish: результируюшая кривая предложения
	:param bids_distribution: вычисленное распределение объема пуска/останова по ступеням
	:param prices: вектор цен
	:return:
	"""
	with open('bids_report.txt', 'a') as file:
		line1 = f'\n<<<\n{info_line}\n'
		line2 = f'исходная кривая предложения : {quantities_start};' \
		        f' результирующая кривая предложения : {quantities_finish}\n'
		line3 = f'распределение объема пусков/остановов по ступеням : {bids_distribution}\n'
		line4 = f'соответствующий вектор цен ступеней : {prices}\n>>>'
		file.writelines([line1, line2, line3, line4])


def get_energy_proposal(
		filename: str = 'bids_distribution.csv'
) -> pd.DataFrame:
	"""
	эта функция считывает данные кривой предложения из csv-файла
	:return:
	"""
	energy_proposal_dataframe = pd.read_csv(
		filepath_or_buffer=filename,
		sep=';',
		encoding='windows-1251',
		parse_dates=['date']
	)

	# here we create dictionary of types in energy proposal
	keys_list = energy_proposal_dataframe.loc[:, 'Type'].unique()
	values_list = range(len(keys_list))
	types_of_energy_proposal = dict(zip(keys_list, values_list))
	# here we add 'type' column with the integer type of the proposal according
	# to types_of_energy_proposal dictionary
	energy_proposal_dataframe['type'] = energy_proposal_dataframe.loc[:, 'Type'].apply(
		lambda x: types_of_energy_proposal[x]
	)
	energy_proposal_dataframe['value'] = energy_proposal_dataframe.loc[:, 'value'].apply(
		lambda x: x.replace(',', '.')
	)
	energy_proposal_dataframe = energy_proposal_dataframe.astype(
		{
			'value': 'float64'
		}
	)
	return energy_proposal_dataframe


def equalize_proposal_matrices(
		proposal_matrices: Tuple[np.ndarray, np.ndarray]
) -> Tuple[Union[np.ndarray, Any], Union[np.ndarray, Any]]:
	"""
	Эта функция получает на вход кортеж из 2 матриц предложения электроэнергии размерности 2*n и 2*m.
	Первая строка каждой матрицы - это вектор цен, 2-я строка - вектор соответствующих объемов предложения.
	В теле функции мы добавляем отсутствующие ступени в матрицу с меньшим количеством столбцов.

	:param proposal_matrices: кортеж с матрицами предложения
	:return: кортеж с расширенными матрицами предложения
	"""
	"""
	Сначала определим базовую матрицу - ту, которая содержит больше ступеней (больше столбцов). Если размерности
	матриц совпадают, то просто берем их по порядку.
	"""
	base_proposal_matrix = np.zeros_like(proposal_matrices[0])
	adjusted_proposal_matrix = np.zeros_like(proposal_matrices[1])
	if proposal_matrices[0].size == proposal_matrices[1].size:
		base_proposal_matrix = proposal_matrices[0]
		adjusted_proposal_matrix = proposal_matrices[1]
	"""
	Если размерности матриц не совпадают, то формируем словарь, ключи в котором соответствуют количеству столбцов
	в матрицах предложений, а значения - самим этим матрицам.
	"""
	proposal_matrices_dictionary = dict(
		zip(map(lambda x: x.size, proposal_matrices), proposal_matrices)
	)
	"""
	В базовую матрицу предложения записываем последний элемент отсортированного словаря, а в матрицу предложения,
	которую требуется расширить - нулевой элемент отсортированного словаря.
	"""
	adjusted_proposal_matrix = proposal_matrices_dictionary[sorted(proposal_matrices_dictionary)[0]]
	base_proposal_matrix = proposal_matrices_dictionary[sorted(proposal_matrices_dictionary)[1]]

	"""
	Теперь определим, какие элементы присутствуют в base_proposal_matrix[0] (что обозначает вектор цен), но
	отсутствуют в adjusted_proposal_matrix[0].
	"""
	new_prices = base_proposal_matrix[
		0, np.invert(np.isin(base_proposal_matrix[0, :], adjusted_proposal_matrix[0, :]))
	]
	"""
	Теперь определим индексы найденных элементов в отсторитрованном векторе adjusted_proposal_matrix[0]
	"""
	new_prices_indices = np.searchsorted(adjusted_proposal_matrix[0, :], new_prices)
	"""
	Теперь проведем итерацию по кортежам (new_prices_indices[i], new_prices[i]); на каждой
	итерации вставим перед new_prices_indices[i]-ым столбцом матрицы adjusted_proposal_matrix новый столбец
	(new_prices[i], 0)
	"""
	index_shift = 0
	for (index, price) in zip(new_prices_indices, new_prices):
		adjusted_proposal_matrix = np.insert(
			adjusted_proposal_matrix,
			index + index_shift,
			[price, adjusted_proposal_matrix[1, index + index_shift]],
			axis=1
		)
		index_shift += 1
	return base_proposal_matrix, adjusted_proposal_matrix



from datetime import datetime
from dateutil.relativedelta import relativedelta

# ##------- Parameters ----------
# # initial_case = "MAY_20_70N_120W_all_year_30Gg"
# # output_path = "MAY_20_case_30Gg_70N_120E/"
# initial_case = "JUL_8_RU_all_year"
# output_path = "JUL_8_30Gg_all_year_RU"
# output_name = ""
# ##-----------------------------

# ##------- Parameters ----------
# # initial_case = "MAY_20_70N_120W_summer_30Gg"
# initial_case = "JUL_8_RU_summer"
# output_path = "JUL_8_30Gg_summer_RU"
# output_name = ""
# ##-----------------------------

# ##------- Parameters ----------
# # initial_case = "MAY_20_70N_120W_winter_30Gg"
# initial_case = "JUL_8_RU_winter"
# output_path = "JUL_8_30Gg_winter_RU"
# output_name = ""
# ##-----------------------------

# ##------- Parameters ----------
# # initial_case = "MAY_20_30N_80W_all_year_30Gg"
# initial_case = "JUL_8_CC_all_year"
# output_path = "JUL_8_30Gg_all_year_CC"
# output_name = ""
# ##-----------------------------

##------- Parameters ----------
initial_case = "JUL_8_control"
output_path = "JUL_8_control"
output_name = ""
##-----------------------------

# text_command = "ncks -v T,O3,date,Z3,bc_a1,bc_a4,FSNTOAC,FSNTC,FSUTOA,FSN200,FSN200C,U,V,OMEGA,FLN200,FLN200C,FLUTC,FLNTC,FLNT,FSNT,Wzm,MASS,bc_a1DTQ,bc_a4DTQ,H2O,NOX,CLOUD,CT_bc_a1,CT_bc_a4,bc_a1WET,bc_a4WET,bc_a1_SRF,bc_a4_SRF "
# text_command_lon =  "ncwa -a lon -v T,O3,date,Z3,bc_a1,bc_a4,FSNTOAC,FSNTC,FSUTOA,FSN200,FSN200C,FLN200,FLN200C,FLUTC,FLNTC,FLNT,FSNT,MASS,bc_a1DTQ,bc_a4DTQ,H2O,NOX,QRL,QRS,QRLC,QRSC,FDL,FDLC,FDS,FDSC,FUS,FUSC,FUL,FULC,U,V,CLOUD,Wzm,OMEGA "

text_command = "ncks -v T,O3,date,Z3,bc_a1,bc_a4,FSNTOAC,FSNTC,FSUTOA,FSN200,FSN200C,U,V,OMEGA,FLN200,FLN200C,FLUTC,FLNTC,FLNT,FSNT,Wzm,MASS,bc_a1DTQ,bc_a4DTQ,H2O,NOX,CLOUD,CT_bc_a1,CT_bc_a4,bc_a1WET,bc_a4WET,bc_a1_SRF,bc_a4_SRF,CLDICE,CLDLIQ,NUMICE,NUMLIQ,IWC,TGCLDIWP,AREI,FICE,NO3,N2O,TROP_P,TROP_Z "
text_command_lon =  "ncwa -a lon -v T,O3,date,Z3,bc_a1,bc_a4,FSNTOAC,FSNTC,FSUTOA,FSN200,FSN200C,FLN200,FLN200C,FLUTC,FLNTC,FLNT,FSNT,MASS,bc_a1DTQ,bc_a4DTQ,H2O,NOX,QRL,QRS,QRLC,QRSC,FDL,FDLC,FDS,FDSC,FUS,FUSC,FUL,FULC,U,V,CLOUD,Wzm,OMEGA,CLDICE,CLDLIQ,NUMICE,NUMLIQ,IWC,TGCLDIWP,AREI,FICE,NO3,N2O,TROP_P,TROP_Z "

# text_command = ""
# text_command_lon =  "ncwa -a lon "


# text_command_lat = "ncwa -w gw -a lat "

save_path = "output_commands.txt"

# initial_case_list = ["JUL_8_control", "JUL_8_CC_all_year", "JUL_8_RU_winter", "JUL_8_RU_summer", "JUL_8_RU_all_year","JUL_8_CC_summer", "JUL_8_CC_winter", "JUL_8_SA_all_year", "JUL_8_SA_summer", "JUL_8_SA_winter", "AUG_13_EQ_summer", "AUG_13_EQ_winter", "AUG_13_EQ_all_year", "AUG_13_UK_all_year", "AUG_13_AU_all_year"]
# output_path_list = ["JUL_8_control", "JUL_8_30Gg_all_year_CC", "JUL_8_30Gg_winter_RU", "JUL_8_30Gg_summer_RU", "JUL_8_30Gg_all_year_RU", "JUL_8_30Gg_summer_CC", "JUL_8_30Gg_winter_CC", "JUL_8_30Gg_all_year_SA", "JUL_8_30Gg_summer_SA", "JUL_8_30Gg_winter_SA", "AUG_13_30Gg_summer_EQ", "AUG_13_30Gg_winter_EQ", "AUG_13_30Gg_all_year_EQ", "AUG_13_30Gg_all_year_UK", "AUG_13_30Gg_all_year_AU"]

# initial_case_list = ["JUL_8_control", "JUL_8_RU_all_year", "JUL_8_RU_summer", "JUL_8_RU_winter", "JUL_8_CC_all_year", "JUL_8_CC_summer", "JUL_8_CC_winter"]
# output_path_list = ["JUL_8_control", "JUL_8_30Gg_all_year_RU", "JUL_8_30Gg_summer_RU",  "JUL_8_30Gg_winter_RU", "JUL_8_30Gg_all_year_CC", "JUL_8_30Gg_summer_CC", "JUL_8_30Gg_winter_CC"]

# initial_case_list = ["AUG_13_UK_all_year", "AUG_13_EQ_all_year", "AUG_13_EQ_summer", "AUG_13_EQ_winter", "AUG_13_AU_all_year", "JUL_8_SA_all_year", "JUL_8_SA_summer", "JUL_8_SA_winter"]
# output_path_list = ["AUG_13_30Gg_all_year_UK", "AUG_13_30Gg_all_year_EQ", "AUG_13_30Gg_summer_EQ", "AUG_13_30Gg_winter_EQ", "AUG_13_30Gg_all_year_AU", "JUL_8_30Gg_all_year_SA", "JUL_8_30Gg_summer_SA", "JUL_8_30Gg_winter_SA"]
# initial_case_list = ["AUG_13_CC_all_year_30Gg_BC_only"]
# output_path_list = ["AUG_13_CC_all_year_30Gg_BC_only"]

# initial_case_list = ["AUG_13_EQ_all_year_20km","AUG_13_EQ_all_year_70km", "AUG_13_EQ_all_year_30km"]
# output_path_list = ["AUG_13_30Gg_all_year_EQ_20km", "AUG_13_30Gg_all_year_EQ_70km", "AUG_13_30Gg_all_year_EQ_30km"]
# initial_case_list = ["JUL_8_control"]
# output_path_list = ["JUL_8_control"]
# initial_case_list = ["AUG_13_AU_all_year_20Gg", "AUG_13_AU_all_year_40Gg", "AUG_13_CC_all_year_20Gg", "AUG_13_CC_all_year_40Gg", "AUG_13_EQ_all_year_20Gg", "AUG_13_EQ_all_year_40Gg"]
# output_path_list = ["AUG_13_20Gg_all_year_AU", "AUG_13_40Gg_all_year_AU","AUG_13_20Gg_all_year_CC", "AUG_13_40Gg_all_year_CC","AUG_13_20Gg_all_year_EQ", "AUG_13_40Gg_all_year_EQ"]

# initial_case_list = ["JUL_8_control", "JUL_8_RU_all_year", "JUL_8_RU_summer", "JUL_8_RU_winter", "JUL_8_CC_all_year", "JUL_8_CC_summer", "JUL_8_CC_winter", "AUG_13_UK_all_year", "AUG_13_EQ_all_year", "AUG_13_EQ_summer", "AUG_13_EQ_winter", "AUG_13_AU_all_year", "JUL_8_SA_all_year", "JUL_8_SA_summer", "JUL_8_SA_winter", "AUG_13_CC_all_year_30Gg_BC_only", "AUG_13_EQ_all_year_20km","AUG_13_EQ_all_year_70km", "AUG_13_EQ_all_year_30km", "AUG_13_AU_all_year_20Gg", "AUG_13_AU_all_year_40Gg", "AUG_13_CC_all_year_20Gg", "AUG_13_CC_all_year_40Gg", "AUG_13_EQ_all_year_20Gg", "AUG_13_EQ_all_year_40Gg"]
# output_path_list = ["JUL_8_control", "JUL_8_30Gg_all_year_RU", "JUL_8_30Gg_summer_RU",  "JUL_8_30Gg_winter_RU", "JUL_8_30Gg_all_year_CC", "JUL_8_30Gg_summer_CC", "JUL_8_30Gg_winter_CC", "AUG_13_30Gg_all_year_UK", "AUG_13_30Gg_all_year_EQ", "AUG_13_30Gg_summer_EQ", "AUG_13_30Gg_winter_EQ", "AUG_13_30Gg_all_year_AU", "JUL_8_30Gg_all_year_SA", "JUL_8_30Gg_summer_SA", "JUL_8_30Gg_winter_SA", "AUG_13_CC_all_year_30Gg_BC_only", "AUG_13_30Gg_all_year_EQ_20km", "AUG_13_30Gg_all_year_EQ_70km", "AUG_13_30Gg_all_year_EQ_30km", "AUG_13_20Gg_all_year_AU", "AUG_13_40Gg_all_year_AU","AUG_13_20Gg_all_year_CC", "AUG_13_40Gg_all_year_CC","AUG_13_20Gg_all_year_EQ", "AUG_13_40Gg_all_year_EQ"]

initial_case_list = ["AUG_13_ID_all_year"]
output_path_list = ["AUG_13_all_year_ID"]
output_name_list = [""]*len(initial_case_list)

with open(save_path, "w") as file:
    for i in range(len(initial_case_list)):
        date_str = "2023-12"
        final_date = "2024-11"
        date = datetime.strptime(date_str, "%Y-%m")
        final_date = datetime.strptime(final_date, "%Y-%m")+ relativedelta(months=1)
        final_date = datetime.strftime(final_date, "%Y-%m")
        while date.strftime("%Y-%m") != final_date:
            date_formatted = date.strftime('%Y-%m')
            initial_file = f"/glade/derecho/scratch/nattanan/archive/{initial_case_list[i]}/atm/hist/{initial_case_list[i]}.cam.h0.{date_formatted}.nc"

            # initial_file = f"/glade/campaign/univ/umic0109/all_output_no_KC/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

            # make_dir = f"mkdir -p /glade/campaign/univ/umic0109/all_output/{output_path_list[i]}{output_name_list[i]}"
            # output_file = f"/glade/campaign/univ/umic0109/all_output/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
            make_dir_lon = f"mkdir -p /glade/campaign/univ/umic0109/output_z_trop_latest/lon/{output_path_list[i]}{output_name_list[i]}"
            output_file_lon = f"/glade/campaign/univ/umic0109/output_z_trop_latest/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

            # make_dir_lat = f"mkdir -p /glade/campaign/univ/umic0109/output_z_trop_latest/lat/{output_path_list[i]}{output_name_list[i]}"
            # output_file_lat = f"/glade/campaign/univ/umic0109/output_z_trop_latest/lat/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
            # text = text_command_lon+initial_file+" "+output_file_lon+"\n"
            # text = make_dir+"\n"+text_command+initial_file+" "+output_file+"\n"+make_dir_lon+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
            text = make_dir_lon+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
            # text = text_command_lon+initial_file+" "+output_file_lon+"\n"
            # text = make_dir_lat+"\n"+text_command_lat+initial_file+" "+output_file_lat+"\n"
            # text = text_command+initial_file+" "+output_file+"\n"
            file.write(text)
            date += relativedelta(months=1)

# initial_case_list = ["AUG_13_AU_all_year"]
# output_path_list = ["AUG_13_30Gg_all_year_AU"]
# output_name_list = [""]*len(initial_case_list)

# with open(save_path, "w") as file:
#     for i in range(len(initial_case_list)):
#         date_str = "2000-12"
#         final_date = "2001-02"
#         date = datetime.strptime(date_str, "%Y-%m")
#         final_date = datetime.strptime(final_date, "%Y-%m")+ relativedelta(months=1)
#         final_date = datetime.strftime(final_date, "%Y-%m")
#         while date.strftime("%Y-%m") != final_date:
#             date_formatted = date.strftime('%Y-%m')
#             initial_file = f"/glade/derecho/scratch/nattanan/archive/{initial_case_list[i]}/atm/hist/{initial_case_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/work/nattanan/derecho_cases/output/{output_path}{output_name}/ncks/{output_name}.cam.h0.{date_formatted}.nc"
#             output_file = f"/glade/campaign/univ/umic0109/output/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file_lon = f"/glade/campaign/univ/umic0109/output/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

#             # text = text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
#             text = text_command+initial_file+" "+output_file+"\n"
#             file.write(text)
#             date += relativedelta(months=1)

# initial_case_list = ["JUL_8_CC_summer", "JUL_8_CC_winter", "JUL_8_SA_all_year", "JUL_8_SA_summer", "JUL_8_SA_winter"]
# output_path_list = ["JUL_8_30Gg_summer_CC", "JUL_8_30Gg_winter_CC", "JUL_8_30Gg_all_year_SA", "JUL_8_30Gg_summer_SA", "JUL_8_30Gg_winter_SA"]
# output_name_list = ["","","","",""]
# # initial_case_list = ["JUL_8_control"]
# # output_path_list = ["JUL_8_control"]
# # output_name_list = [""]

# with open(save_path, "w") as file:
#     for i in range(len(initial_case_list)):
#         date_str = "2016-12"
#         final_date = "2019-11"
#         date = datetime.strptime(date_str, "%Y-%m")
#         final_date = datetime.strptime(final_date, "%Y-%m")+ relativedelta(months=1)
#         final_date = datetime.strftime(final_date, "%Y-%m")
#         while date.strftime("%Y-%m") != final_date:
#             date_formatted = date.strftime('%Y-%m')
#             initial_file = f"/glade/derecho/scratch/nattanan/archive/{initial_case_list[i]}/atm/hist/{initial_case_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/work/nattanan/derecho_cases/output/{output_path}{output_name}/ncks/{output_name}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/univ/umic0109/output/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
#             output_file_lon = f"/glade/campaign/univ/umic0109/output/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

#             text = text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"
#             file.write(text)
#             date += relativedelta(months=1)

# initial_case_list = ["AUG_13_EQ_summer", "AUG_13_EQ_winter", "AUG_13_EQ_all_year", "AUG_13_UK_all_year"]
# output_path_list = ["AUG_13_30Gg_summer_EQ", "AUG_13_30Gg_winter_EQ", "AUG_13_30Gg_all_year_EQ", "AUG_13_30Gg_all_year_UK"]

# # initial_case_list = ["AUG_13_UK_all_year"]
# # output_path_list = ["AUG_13_30Gg_all_year_UK"]
# # output_name_list = [""]
# # initial_case_list = ["JUL_8_control"]
# # output_path_list = ["JUL_8_control"]
# output_name_list = ["","","",""]

# with open(save_path, "w") as file:
#     for i in range(len(initial_case_list)):
#         date_str = "2010-12"
#         final_date = "2019-11"
#         date = datetime.strptime(date_str, "%Y-%m")
#         final_date = datetime.strptime(final_date, "%Y-%m")+ relativedelta(months=1)
#         final_date = datetime.strftime(final_date, "%Y-%m")
#         while date.strftime("%Y-%m") != final_date:
#             date_formatted = date.strftime('%Y-%m')
#             initial_file = f"/glade/derecho/scratch/nattanan/archive/{initial_case_list[i]}/atm/hist/{initial_case_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/work/nattanan/derecho_cases/output/{output_path}{output_name}/ncks/{output_name}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/univ/umic0109/output/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
#             output_file_lon = f"/glade/campaign/univ/umic0109/output/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

#             text = text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"
#             file.write(text)
#             date += relativedelta(months=1)

# initial_case_list = ["AUG_13_AU_all_year"]
# output_path_list = ["AUG_13_30Gg_all_year_AU"]
# output_name_list = [""]

# with open(save_path, "w") as file:
#     for i in range(len(initial_case_list)):
#         date_str = "2008-12"
#         final_date = "2019-11"
#         date = datetime.strptime(date_str, "%Y-%m")
#         final_date = datetime.strptime(final_date, "%Y-%m")+ relativedelta(months=1)
#         final_date = datetime.strftime(final_date, "%Y-%m")
#         while date.strftime("%Y-%m") != final_date:
#             date_formatted = date.strftime('%Y-%m')
#             initial_file = f"/glade/derecho/scratch/nattanan/archive/{initial_case_list[i]}/atm/hist/{initial_case_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/work/nattanan/derecho_cases/output/{output_path}{output_name}/ncks/{output_name}.cam.h0.{date_formatted}.nc"
#             output_file = f"/glade/campaign/univ/umic0109/output/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
#             output_file_lon = f"/glade/campaign/univ/umic0109/output/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

#             text = text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"
#             file.write(text)
#             date += relativedelta(months=1)

# initial_case_list = ["AUG_13_EQ_all_year_20km","AUG_13_EQ_all_year_70km"]
# output_path_list = ["AUG_13_30Gg_all_year_EQ_20km", "AUG_13_30Gg_all_year_EQ_70km"]
# output_name_list = ["",""]

# with open(save_path, "w") as file:
#     for i in range(len(initial_case_list)):
#         date_str = "2000-12"
#         final_date = "2019-11"
#         date = datetime.strptime(date_str, "%Y-%m")
#         final_date = datetime.strptime(final_date, "%Y-%m")+ relativedelta(months=1)
#         final_date = datetime.strftime(final_date, "%Y-%m")
#         while date.strftime("%Y-%m") != final_date:
#             date_formatted = date.strftime('%Y-%m')
#             initial_file = f"/glade/derecho/scratch/nattanan/archive/{initial_case_list[i]}/atm/hist/{initial_case_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/work/nattanan/derecho_cases/output/{output_path}{output_name}/ncks/{output_name}.cam.h0.{date_formatted}.nc"
#             output_file = f"/glade/campaign/univ/umic0109/output/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file_lon = f"/glade/campaign/univ/umic0109/output/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

#             # text = text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
#             text = text_command+initial_file+" "+output_file+"\n"
#             file.write(text)
#             date += relativedelta(months=1)

# initial_case_list = ["AUG_13_EQ_all_year_30km"]
# output_path_list = ["AUG_13_30Gg_all_year_EQ_30km"]
# output_name_list = ["",""]

# with open(save_path, "w") as file:
#     for i in range(len(initial_case_list)):
#         date_str = "2000-12"
#         final_date = "2019-11"
#         date = datetime.strptime(date_str, "%Y-%m")
#         final_date = datetime.strptime(final_date, "%Y-%m")+ relativedelta(months=1)
#         final_date = datetime.strftime(final_date, "%Y-%m")
#         while date.strftime("%Y-%m") != final_date:
#             date_formatted = date.strftime('%Y-%m')
#             initial_file = f"/glade/derecho/scratch/nattanan/archive/{initial_case_list[i]}/atm/hist/{initial_case_list[i]}.cam.h0.{date_formatted}.nc"
#             # output_file = f"/glade/campaign/work/nattanan/derecho_cases/output/{output_path}{output_name}/ncks/{output_name}.cam.h0.{date_formatted}.nc"
#             output_file = f"/glade/campaign/univ/umic0109/output/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"
#             output_file_lon = f"/glade/campaign/univ/umic0109/output/lon/{output_path_list[i]}{output_name_list[i]}/{output_path_list[i]}.cam.h0.{date_formatted}.nc"

#             # text = text_command_lon+initial_file+" "+output_file_lon+"\n"
#             # text = text_command+initial_file+" "+output_file+"\n"+text_command_lon+initial_file+" "+output_file_lon+"\n"
#             text = text_command+initial_file+" "+output_file+"\n"
#             file.write(text)
#             date += relativedelta(months=1)